import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class Scorer_ANN(nn.Module):
    def __init__(self, num_channel, in_length, num_seg_y, num_seg_x, dropout_prob=0.1):
        super(Scorer_ANN, self).__init__()

        self.num_channel = num_channel
        self.in_length = in_length
        self.num_seg_y = num_seg_y
        self.num_seg_x = num_seg_x
        self.kernel_nms = 3

        self.deployed = False
        self.dropout = nn.Dropout(p=dropout_prob)

        # Feature extraction without max pooling
        self.conv1 = nn.Conv2d(in_length, num_channel, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=2, dilation=2)  

        # Scalar score output
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Outputs shape (B*N, C, 1, 1)
        self.fc1 = nn.Linear(num_channel, 1)  # Final scalar score

    def forward(self, x):  
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # Flatten batch

        x = F.leaky_relu(self.conv1(x))  
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x))  
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x))  
        x = self.dropout(x)

        x = self.global_avg_pool(x)  # Reduce to (B*N, C, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B*N, C)
        x = self.fc1(x)  # Linear to scalar (B*N, 1)

        x = x.view(B,1,self.num_seg_y,self.num_seg_x) # B x 1 x n_seg_y x n_seg_x

        # Apply max pooling to find local maxima
        pooled_map = F.max_pool2d(x, kernel_size=self.kernel_nms, stride=1, padding=self.kernel_nms // 2)

        # Create a mask where the score map equals the pooled map (local maxima positions)
        maxima_mask = (x == pooled_map).float()
        min_value = torch.amin(x,dim=(1,2,3),keepdim=True)
        x = (x - min_value)
        if self.deployed:
            non_maxima_mask = (x != pooled_map).float()
            x = x * maxima_mask - 0.01 * x * non_maxima_mask
        else:
            x = x * maxima_mask
        x + min_value

        return x.view(-1) 
    
class Feature_ANN(nn.Module):
    def __init__(self, hidden_1, hidden_2, data_dim, in_length, seg_size, use_norm=True, num_channel=64, dropout_prob=0.1,
    num_conv_layers = 3, num_fc_layers = 2):
        super(Feature_ANN, self).__init__()

        self.data_dim = data_dim + 3
        self.in_length = in_length
        self.seg_size = seg_size
        self.num_channel = num_channel  # Increased channels
        self.use_norm = use_norm  # Flag to enable/disable normalization

        in_channels = self.in_length + 1
        out_channels = num_channel
        conv_layers = []
        # Convolutional layers
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.GroupNorm(num_groups=4, num_channels=out_channels) if use_norm else nn.Identity())
            conv_layers.append(nn.LeakyReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(kernel_size=2))  # Pooling inside
            in_channels = out_channels
            out_channels *= 2  # Doubling channels each layer

        conv_layers.append(nn.AdaptiveAvgPool2d(1))  # Global average pooling at the end
        self.conv_layers = nn.Sequential(*conv_layers)

        # Fully connected layers
        fc_layers = []
        in_features = in_channels

        for i in range(num_fc_layers - 1): 
            fc_layers.append(nn.Linear(in_features, in_features))
            fc_layers.append(nn.LeakyReLU(inplace=True)) 
            fc_layers.append(nn.Dropout(p=dropout_prob)) 

        fc_layers.append(nn.Linear(in_features, self.data_dim))
        self.fc_layers = nn.Sequential(*fc_layers)

    def add_noise_channel(self, x):
        # Adds a random noise channel
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), dtype=torch.float32).to(x.device)
        return torch.cat([x, noise], 1)

    def forward(self, x):  # N x 1 x 28 x 28
        x = self.add_noise_channel(x)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x) 

        x1 = torch.tanh(x[:, :2]) * self.seg_size / 2
        x2 = torch.sigmoid(x[:, 2:])
        x = torch.cat([x1, x2], 1)

        return x


class ArgMaxSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, K, device):
        # x must be shape C x d, where d is the number of elements
        # N is the number of noise samples
        # K is the number of elements to take out of the stack
        # sigma is the std of the noise

        C = x.size(0)
        d = x.size(1)

        # Sort
        x, x_ind = torch.sort(x, dim=1, descending=True)
        _, x_ind_inv = torch.sort(x_ind, dim=1)

        _, arg_locs = torch.topk(x.view(C, d), k=K, dim=1)  # Output is of size C x K. Actual maxima not needed
        arg_locs, _ = torch.sort(arg_locs, dim=1)

        one_hot = torch.zeros(C, d, K).to(device)

        dim_0 = torch.arange(C).view(C, 1).expand(C, K).to(device)
        dim_1 = arg_locs
        dim_2 = torch.arange(K).view( 1, K).expand(C, K).to(device)

        one_hot[dim_0, dim_1, dim_2] = 1  # one_hot of size C x d x K

        x_expanded     = x_ind_inv.unsqueeze(2).expand(-1,-1,K)
        one_hot     = torch.gather(one_hot,1,x_expanded)

        return one_hot

# Jacobian for each K
class SoftArgMaxSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,N,K,sigma,device):
        # x must be shape C x d, where d is the number of elements
        # N is the number of noise samples
        # K is the number of elements to take out of the stack
        # sigma is the std of the noise
        
        C            = x.size(0)
        d            = x.size(1)
        
        # Sort
        x,x_ind      = torch.sort(x,dim=1,descending=True)
        _,x_ind_inv  = torch.sort(x_ind,dim=1)
        x_ind_inv    = x_ind_inv.to(device)
        
        noise        = torch.randn(C,N,d).to(device)
        noisy_vector = noise*sigma + x.view(C,1,d)
        
        _,arg_locs = torch.topk(noisy_vector,k=K,dim=2) # Output is of size C x N x K. Actual maxima not needed
        arg_locs,_ = torch.sort(arg_locs,dim=2)
        
        one_hot = torch.zeros(C,N,d,K).to(device)
        
        dim_0   = torch.arange(C).view(C,1,1).expand(C,N,K).to(device)
        dim_1   = torch.arange(N).view(1,N,1).expand(C,N,K).to(device)
        dim_2   = arg_locs
        dim_3   = torch.arange(K).view(1,1,K).expand(C,N,K).to(device)
        
        one_hot[dim_0,dim_1,dim_2,dim_3] = 1 # one_hot of size C x N x d x K
                      
        y_eps_star     = torch.mean(one_hot,1) # Is C x d x K
                     
        jac_y_eps_star = torch.matmul(one_hot.permute(0,3,2,1),noise.unsqueeze(1))/((sigma+1e-10) * N) # is C x d x d
                
        # Unsort again
        x_expanded     = x_ind_inv.unsqueeze(2).expand(-1,-1,K)
        y_eps_star     = torch.gather(y_eps_star,1,x_expanded)
                
        x_expanded     = x_ind_inv.unsqueeze(1).unsqueeze(3).expand(-1,K,-1,d)
        jac_y_eps_star = torch.gather(jac_y_eps_star,2,x_expanded)
        x_expanded     = x_ind_inv.unsqueeze(1).unsqueeze(2).expand(-1,K,d,-1)
        jac_y_eps_star = torch.gather(jac_y_eps_star,3,x_expanded)
        
        ctx.save_for_backward(jac_y_eps_star)
        
        return y_eps_star
    
    @staticmethod
    def backward(ctx,grad_output):        
        jac_y_eps_star, = ctx.saved_tensors
        
        grad_input = torch.matmul(jac_y_eps_star.permute(0,1,3,2),grad_output.permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        grad_input = torch.sum(grad_input,1)
                        
        return grad_input, None, None, None, None

class ListGen_ANN(nn.Module):
    def __init__(self,sigma,K=2,data_dim=1,input_size: int or (int,int)=64,seg_size=28,seg_shift=6,im_dim=1, 
    num_channel_scorer = 24, num_channel=64, num_conv_layers = 4, num_fc_layers = 3, n_pad = 3):
        super(ListGen_ANN,self).__init__()
        if isinstance(input_size, int):
            self.input_size = (input_size+2*n_pad,input_size+2*n_pad)
        else:
            self.input_size  = tuple(x + 2*n_pad for x in input_size)
        self.seg_size    = seg_size
        self.seg_shift   = seg_shift
        self.n_pad       = n_pad
        self.N           = 250 # Tune this
        self.sigma       = sigma
        self.K           = K
        
        self.net_dim     = num_channel_scorer
        self.im_dim      = im_dim
        
        self.data_dim    = data_dim # Number of eta dimensions (not loc and alpha)
        
        self.num_seg_y     = math.ceil((self.input_size[0]-self.seg_size)/self.seg_shift)+1
        self.num_seg_x = math.ceil((self.input_size[1] - self.seg_size) / self.seg_shift) + 1

        self.pad_h = (self.num_seg_y * self.seg_shift + self.seg_size - self.input_size[0]) % self.seg_shift
        self.pad_w = (self.num_seg_x * self.seg_shift + self.seg_size - self.input_size[1]) % self.seg_shift
        # Scorer network s
        self.scorer_ann = Scorer_ANN(self.net_dim,self.im_dim,self.num_seg_y,self.num_seg_x)

        # Feature network f
        self.feature_ann     = Feature_ANN(self.net_dim,self.net_dim,self.data_dim,self.im_dim,self.seg_shift, num_channel=num_channel, num_conv_layers=num_conv_layers, num_fc_layers=num_fc_layers)
        
        self.deployed = False
        self.hard_max = False
        self.lr          = 1
        
        # Get the shifts right
        vec_1   = torch.arange(self.num_seg_x).repeat(self.num_seg_y)*self.seg_shift
        vec_0,_ = torch.sort(torch.arange(self.num_seg_y).repeat(self.num_seg_x)*self.seg_shift)
        self.register_buffer('shifts', torch.zeros((vec_0.shape[0],2),dtype=int))
        self.shifts[:,0] = vec_0
        self.shifts[:,1] = vec_1
        
    def update_lr(self,lr):
        self.lr = lr

    def state_dict(self, *args, **kwargs):
        # Get the state dict without the buffer 'shifts'
        state_dict = super(ListGen_ANN, self).state_dict(*args, **kwargs)
        state_dict.pop('shifts', None)  # Remove 'shifts' from the state dict
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Load the state dict while ignoring 'shifts'
        state_dict.pop('shifts', None)  # Remove 'shifts' from the state dict before loading
        super(ListGen_ANN, self).load_state_dict(state_dict, strict=strict)
        
    def segmentation(self,x):   
        # Cut data into segments (do it for each axis once)
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = x.unfold(2,self.seg_size,self.seg_shift)
        x = x.unfold(3,self.seg_size,self.seg_shift)
        
        x = x.permute(0,2,3,4,5,1)
        
        dim = x.shape[1]
        
        # Reshape two have only one axis
        x = x.contiguous().view(x.size(0),self.num_seg_y*self.num_seg_x,self.seg_size,self.seg_size,x.size(-1))
        
        x = x.permute(0,1,4,2,3) # C * seg_shift^2 x 1 x seg_size x seg_size
        return x, dim
        
    def patching(self,x,softmax):
        # Create the patched images (which are means) 
        x_large       = x.unsqueeze(-1)
        softmax_large = softmax.unsqueeze(2).unsqueeze(3).unsqueeze(4)        
        
        patches       = (x_large * softmax_large).sum(dim=1)
        # patches shape: C x channel x seg_size x seg_size x K
        
        patches       = patches.permute(0,4,1,2,3)
        patches       = patches.contiguous().view(-1,x.size(2),self.seg_size,self.seg_size)
        # patches shape: C*K x channel x seg_size x seg_size
        return patches
    def get_shifts(self, softmax):
        softmax_shifts = softmax.unsqueeze(2)
        shifts = self.shifts.unsqueeze(0).unsqueeze(3)

        shifts = softmax_shifts * shifts
        shifts = torch.sum(shifts,1).permute(0,2,1)
                
        return shifts
        
    def permuting(self,output_list):
        permutations = torch.stack([torch.randperm(self.K) for _ in range(output_list.size(0))],dim=0).to(next(self.parameters()).device)
        output_list  = output_list.gather(1,permutations.unsqueeze(-1).expand(-1,-1,output_list.size(2)))
        
        return output_list
        
    def forward(self,x,softmax=None):    
        
        x = F.pad(x, (self.n_pad, self.n_pad, self.n_pad, self.n_pad), mode='constant', value=0)
        x, dim      = self.segmentation(x)

        predictions = self.scorer_ann(x)

        predictions = predictions.view(-1, self.num_seg_y * self.num_seg_x)
        x = x.view(-1, self.num_seg_y * self.num_seg_x, x.shape[2], self.seg_size, self.seg_size)

        if self.deployed:
            softmax = ArgMaxSingle.apply(predictions, self.K, predictions.device)
            softmax = torch.sum(softmax,dim=-1)
            patches = x[softmax.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x) == 1].view(-1, x.size(2),
                                                                                                  x.size(3), x.size(4))
            shifts = self.shifts.expand(softmax.shape[0], softmax.shape[1], 2)
            shifts = shifts[softmax.unsqueeze(-1).expand_as(shifts) == 1].view(-1,self.K,2)

        else:
            if softmax is None:
                # This way it is possible, to actually skip the softmax entirely for the list cycle loss
                softmax = SoftArgMaxSingle.apply(predictions, self.N, self.K, self.sigma * self.lr, predictions.device)
            patches = self.patching(x,softmax)
            shifts = self.get_shifts(softmax)
        
        output_list    = self.feature_ann(patches).view(-1,self.K,self.data_dim+3)
        output_list    = torch.cat([output_list[:,:,:2]+shifts+self.seg_size/2-self.n_pad,output_list[:,:,2:]],2)
        
        # shape: C x K x (data_dim + 3)
        
        # Also return the predictions, as this is needed for the list cycle loss        
        return output_list, predictions
