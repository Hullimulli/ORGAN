import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class up_conv_block(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_mid: int, n_channels_out: int, up_conv_bool=True, use_norm=True):
        super(up_conv_block, self).__init__()
        self.use_norm = use_norm
        self.uppool = nn.MaxUnpool2d(2, stride=2)
        self.conv_in = nn.Conv2d(n_channels_in, n_channels_mid, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, n_channels_mid) if use_norm else nn.Identity()

        self.conv_out = nn.Conv2d(n_channels_mid, n_channels_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, n_channels_out) if use_norm else nn.Identity()

        self.up_conv_bool = up_conv_bool

    def forward(self, x, x_res, ind_up=None):
        x = self.norm1(self.conv_in(torch.cat((x, x_res), dim=1)))
        x = F.leaky_relu(x)
        x = self.norm2(self.conv_out(x))
        x = F.leaky_relu(x)

        if self.up_conv_bool:
            x = self.uppool(x, indices=ind_up)

        return x


class de_conv_block(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_mid: int, n_channels_out: int, de_conv_bool=True, up_conv_bool=False, use_norm=True):
        super(de_conv_block, self).__init__()
        self.use_norm = use_norm
        self.pool = nn.MaxPool2d(2,stride=2,return_indices=True)
        self.uppool = nn.MaxUnpool2d(2, stride=2)

        self.conv_in = nn.Conv2d(n_channels_in, n_channels_mid, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, n_channels_mid) if use_norm else nn.Identity()

        self.conv_out = nn.Conv2d(n_channels_mid, n_channels_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, n_channels_out) if use_norm else nn.Identity()

        self.up_conv_bool = up_conv_bool
        self.de_conv_bool = de_conv_bool

    def forward(self, x, ind_up=None):
        x = self.norm1(self.conv_in(x))
        x = F.leaky_relu(x)
        x_res = self.norm2(self.conv_out(x))
        x_res = F.leaky_relu(x_res)

        if self.up_conv_bool:
            return self.uppool(x_res, indices=ind_up)
        elif self.de_conv_bool:
            x, idx = self.pool(x_res)
            return x, x_res, idx
        else:
            return x_res

class ImGen_ANN(nn.Module):
    # TODO: Check CycleGAN Generator Network Architecture
    def __init__(self,data_dim=1,out_dim=1,noise_size: int or (int,int)=16,im_size: int or (int,int)=64,data_smooth=2.5,transformSpace=True, use_norm=True,
    num_channel=64, num_output_layers=3, num_layers=3, n_pad=3):
        super(ImGen_ANN,self).__init__()
        if isinstance(im_size,int):
            self.im_size_y      = im_size+n_pad*2
            self.im_size_x      = im_size+n_pad*2
        else:
            self.im_size_y      = im_size[0]+n_pad*2
            self.im_size_x      = im_size[1]+n_pad*2
        if isinstance(noise_size, int):
            self.noise_size_y   = noise_size
            self.noise_size_x = noise_size
        else:
            self.noise_size_y   = noise_size[0]
            self.noise_size_x = noise_size[1]
        self.n_pad = n_pad
        self.data_dim     = data_dim    # Number of eta dimensions (not loc and alpha)
        self.data_smooth  = data_smooth # Number of pixels to smooth to allow for gradients
        self.out_dim      = out_dim

        self.deployed = False
        self.num_channel  = num_channel #16
        self.num_layers = num_layers
        num_output_layers = num_output_layers

        self.lr           = 1
        self.epsilon = 0.005

        grad = np.zeros((self.im_size_y,self.im_size_x,2))
        for i in range(self.im_size_y):
            grad[i,:,0]   = i
        for i in range(self.im_size_x):
            grad[:,i,1]   = i
        self.register_buffer('grad', torch.from_numpy(grad).float().unsqueeze(0).unsqueeze(1))
        self.gauss_scale  = 1/np.sqrt(2*np.pi)
        self.transformSpace = transformSpace
        # Grad shape: 1 x 1 x im_size x im_size x  2

        self.encoders = nn.ModuleList()
        in_channels = self.data_dim + 1 + int(self.transformSpace)
        out_channels = self.num_channel

        for i in range(num_layers):
            self.encoders.append(de_conv_block(in_channels, out_channels, out_channels, use_norm=use_norm))
            in_channels = out_channels
            out_channels *= 2 
        
        # Center block
        self.center = de_conv_block(in_channels, out_channels, in_channels, de_conv_bool=False, up_conv_bool=True, use_norm=use_norm)
        out_channels = in_channels
        self.decoders = nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = out_channels * 2
            mid_channels = in_channels // 2
            out_channels = mid_channels // 2
            self.decoders.append(up_conv_block(in_channels, mid_channels, out_channels, use_norm=use_norm))
        in_channels = out_channels * 2
        out_channels = out_channels
        self.decoders.append(up_conv_block(in_channels, out_channels, out_channels, use_norm=use_norm, up_conv_bool=False))
        self.output_layers = nn.ModuleList()
        for i in range(num_output_layers - 1):
            self.output_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))  # Hidden output layers
        # Final output layer
        self.output_layers.append(nn.Conv2d(out_channels, out_dim, kernel_size=1))  # Last output layer

    def state_dict(self, *args, **kwargs):
        state_dict = super(ImGen_ANN, self).state_dict(*args, **kwargs)
        state_dict.pop('grad', None)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        state_dict.pop('grad', None) 
        super(ImGen_ANN, self).load_state_dict(state_dict, strict=strict)

    def update_lr(self,lr):
        self.lr = lr

    def transform_to_image(self,x):
        rest = x[:, :, 3:]
        alpha = x[:,:,2]  # C x K
        pos_x = x[:,:,0]+self.n_pad  # C x K
        pos_y = x[:,:,1]+self.n_pad  # C x K

        pos_x = pos_x.unsqueeze(2).unsqueeze(3)
        pos_y = pos_y.unsqueeze(2).unsqueeze(3)

        loc_x = torch.exp(-torch.square(pos_x-self.grad[:,:,:,:,0])/(2*(self.data_smooth)**2))
        loc_y = torch.exp(-torch.square(pos_y-self.grad[:,:,:,:,1])/(2*(self.data_smooth)**2))

        loc   = loc_x * loc_y # C x K x im_size x im_size
        loc   = loc.unsqueeze(4)

        if self.transformSpace:
            # Begin of part to transform to hypersphere
            factor = torch.tensor(1).to(x.device)
            rest   = rest * torch.pi
            sin_cumprod = torch.cat(
                [torch.ones_like(rest[:, :, :1]), torch.cumprod(torch.sin(rest), dim=2)], dim=2
            )  # Prepend 1 for correct accumulation
    
            rest = torch.cat([torch.cos(rest), torch.ones_like(rest[:, :, :1])], dim=2)  # Cosine terms + final 1
            rest *= sin_cumprod  # Apply cumulative sine factors
            # End of part to transform to hypersphere

        rest  = rest.unsqueeze(2).unsqueeze(3)
        alpha = alpha.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        data = loc * rest * alpha
        return torch.sum(data,1).permute(0,3,1,2)

    def create_noise_vector(self,x):
        # Returned value has size N x 1 x im_size x im_size
        #TODO Check distribution
        noise = torch.randn(x.size(0),1,self.noise_size_y,self.noise_size_x,dtype=torch.float32).to(x.device)
        return F.interpolate(noise,size=(self.im_size_y,self.im_size_x),mode='bilinear')

    def transform(self,x):
        # Transforms a list to an image
        # list:  C x K x (data_dim+3)
        # Image: C x (data_dim+2) x im_size x im_size
        data  = self.transform_to_image(x)
        noise = self.create_noise_vector(x)

        transformed = torch.cat([data,noise],1)

        return transformed

    def apply_network(self, x):
        original_size = x.size()[2:]  # Save (H, W)
        padding = 2**self.num_layers
        x = F.pad(x, (0, (padding - x.size(3) % padding) % padding, 0, (padding - x.size(2) % padding) % padding))  # Ensure divisibility

        skip_connections = []  # Store encoder outputs
        indices = []  # Store max-pooling indices

        # Encoder forward pass
        for encoder in self.encoders:
            x, x_res, idx = encoder(x)
            skip_connections.append(x_res)
            indices.append(idx)

        # Center block
        x = self.center(x, indices[-1])

        for i, decoder in enumerate(self.decoders):
            if i == len(self.decoders) - 1:
                # Last decoder does not need pooling indices
                x = decoder(x, skip_connections[-(i + 1)])
            else:
                x = decoder(x, skip_connections[-(i + 1)], indices[-(i + 2)])

        # Crop back to original size
        x = x[:, :, :original_size[0], :original_size[1]]

        # Pass through multiple output layers
        for output_layer in self.output_layers:
            x = output_layer(x)

        return x

    def forward(self,x): # Should be getting a list. List has the shape: C x K x (data_dim+3)
        # This blur makes it more difficult to differentiate between the two paths
        #x           = x + torch.randn(*x.shape).to(x.device)*0.1*self.lr

        transformed = self.transform(x)
        out         = self.apply_network(transformed)
        if self.n_pad!=0:
            out = out[:, :, self.n_pad:-self.n_pad, self.n_pad:-self.n_pad]
        return out