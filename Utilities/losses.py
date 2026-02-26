import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from torch import Tensor

class MSE_Loss(nn.Module):
    # Scale factor is for comparison with the MAE loss
    def forward(self, X: Tensor, Y: Tensor, factor: float = 2.5) -> Tensor:
        return factor*torch.mean((X - Y) ** 2)
class MAE_Loss(nn.Module):
    def forward(self, X: Tensor, Y: Tensor, factor: float = 1) -> Tensor:
        return factor*torch.mean(torch.abs(X - Y))
class SSIM_Loss(SSIM):

    def forward(self, X: Tensor, Y: Tensor, factor: float = 0.25) -> Tensor:
        return (1 - ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        ))*factor

class MS_SSIM_Loss(MS_SSIM):

    def forward(self, X: Tensor, Y: Tensor, factor: float = 0.25) -> Tensor:
        return (1 - ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        ))*factor

class SSIM_YUV_Loss(nn.Module):
    # Only works for RGB, use SSIM for grayscale
    def __init__(self, lambda_mae=3.0):
        super().__init__()
        self.lambda_mae = lambda_mae    # Weight for MAE on U/V
        self.ssim_loss = SSIM_Loss(data_range=1,size_average=True,channel=1)

    def rgb_to_yuv(self, img: torch.Tensor) -> torch.Tensor:
        """ Converts an RGB image to YUV. Assumes input shape (N, 3, H, W). """
        
        # Transformation matrix
        mat = torch.tensor(
            [[0.299, 0.587, 0.114], 
            [-0.14713, -0.28886, 0.436], 
            [0.615, -0.51499, -0.10001]], 
            dtype=img.dtype, device=img.device
        )

        # Efficient batch matrix multiplication
        yuv = torch.einsum("nchw,mc->nmhw", img, mat)

        return yuv  # Shape remains (N, 3, H, W)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, factor: float = 0.25) -> torch.Tensor:
        """ Computes loss using SSIM on Y and MAE on U/V. """
        yuv_x = self.rgb_to_yuv(X)  # Convert RGB → YUV
        yuv_y = self.rgb_to_yuv(Y)

        # Compute SSIM on Y channel only
        loss_ssim_y = self.ssim_loss(yuv_x[:, 0:1, :, :], yuv_y[:, 0:1, :, :])

        # Compute MAE on U and V channels
        loss_mae_uv = F.l1_loss(yuv_x[:, 1:, :, :], yuv_y[:, 1:, :, :])

        # Weighted sum of losses
        return factor * (loss_ssim_y + self.lambda_mae * loss_mae_uv) / 2

class SSIM_Canny_YUV_Loss(nn.Module):
    def __init__(self, lambda_mae=1.0, lambda_canny=0.5):
        super().__init__()
        self.lambda_mae = lambda_mae  # Weight for MAE on U/V
        self.lambda_canny = lambda_canny  # Weight for Canny edge loss
        self.ssim_loss = SSIM_Loss(data_range=1, size_average=True, channel=1)

        # Sobel kernels for Canny approximation
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,3,3)
        self.sobel_y = self.sobel_x.transpose(2, 3)  # Rotate for Y-direction

    def rgb_to_yuv(self, img: torch.Tensor) -> torch.Tensor:
        """ Converts an RGB image to YUV. Assumes input shape (N, 3, H, W). """
        mat = torch.tensor(
            [[0.299, 0.587, 0.114], 
            [-0.14713, -0.28886, 0.436], 
            [0.615, -0.51499, -0.10001]], 
            dtype=img.dtype, device=img.device
        )
        return torch.einsum("nchw,mc->nmhw", img, mat)

    def canny_edge(self, img: torch.Tensor) -> torch.Tensor:
        """ Approximates Canny edges using Sobel filters on grayscale (Y channel). """
        sobel_x = F.conv2d(img, self.sobel_x.to(img.device), padding=1)
        sobel_y = F.conv2d(img, self.sobel_y.to(img.device), padding=1)
        edges = torch.sqrt(sobel_x**2 + sobel_y**2)  # Gradient magnitude
        return edges

    def forward(self, X: torch.Tensor, Y: torch.Tensor, factor: float = 0.25) -> torch.Tensor:
        """ Computes loss using SSIM on Y, MAE on U/V, and Canny edge loss. """
        yuv_x = self.rgb_to_yuv(X)  # Convert RGB → YUV
        yuv_y = self.rgb_to_yuv(Y)

        # Compute SSIM on Y channel
        loss_ssim_y = self.ssim_loss(yuv_x[:, 0:1, :, :], yuv_y[:, 0:1, :, :])

        # Compute MAE on U and V channels
        loss_mae_uv = F.l1_loss(yuv_x[:, 1:, :, :], yuv_y[:, 1:, :, :])

        # Compute Canny edge loss on Y channel
        edges_x = self.canny_edge(yuv_x[:, 0:1, :, :])
        edges_y = self.canny_edge(yuv_y[:, 0:1, :, :])
        loss_canny = F.l1_loss(edges_x, edges_y)

        # Weighted sum of losses
        return factor * (loss_ssim_y + self.lambda_mae * loss_mae_uv + self.lambda_canny * loss_canny) / 3

class FullLoss():
    def __init__(
            self,
            lst_dim: int,
            seg_size,
            im_size,
            lambda_cyc_img: float = 25,
            lambda_cyc_lst: float = 5,
            lambda_dis_lst: float = 1,
            lambda_dis_img: float = 1,
            cyc_loss_func = SSIM_Loss(data_range=1,size_average=True),
            lambda_loc: float = 5.0,
            lambda_pres: float = 15.0,
            lambda_entropy: float = 0.0,
            lambda_gp: float = 0,
            lambda_lst_prior: float = 1.0,
            lambda_im_prior: float = 1.0,
            n_pad: int = 3
    ):  
        self.seg_size = seg_size
        self.im_size = im_size
        self.lambda_cyc_img = lambda_cyc_img
        self.lambda_cyc_lst = lambda_cyc_lst
        self.lambda_dis_lst = lambda_dis_lst
        self.lambda_dis_img = lambda_dis_img
        self.lambda_entropy = lambda_entropy
        self.cyc_loss_func = cyc_loss_func
        self.lambda_loc = lambda_loc
        self.lambda_pres = lambda_pres
        self.lambda_gp = lambda_gp
        self.lambda_lst_prior = lambda_lst_prior
        self.lambda_im_prior = lambda_im_prior
        self.lst_elem_weight = torch.ones((1,1,1,3+lst_dim))
        self.lst_elem_weight[..., 3:] = 1 / lst_dim
        self.n_pad = n_pad

    def im_cyc_loss(self,im_gt,im_pred):
        return self.cyc_loss_func(im_gt,im_pred)

    def get_cost_matrix_(self,list_gt,list_pred):
        list_gt_   = list_gt[:,:,:2]/self.im_size # Make locations between 0 and 1
        list_pred_ = list_pred[:,:,:2]/self.im_size # Make locations between 0 and 1

        list_gt   = torch.cat([list_gt_,  list_gt[:,:,2:]],  2)
        list_pred = torch.cat([list_pred_,list_pred[:,:,2:]],2)

        list_gt   =   list_gt.unsqueeze(2)
        list_pred = list_pred.unsqueeze(1)

        alphas_gt   =   list_gt[:,:,:,2]
        alphas_pred = list_pred[:,:,:,2]

        all_loss = torch.square(list_gt - list_pred) * alphas_gt[...,None]
        all_loss[...,2] += (self.lambda_pres - alphas_gt) * torch.square(alphas_gt - alphas_pred)

        return all_loss

    @staticmethod
    def find_closest(shifts, list_real, seg_size, padding=0):
        def k_hot_encoding(tensor, N):
            # Transform from: C x K
            # to:             C x seg_shift^2 x K
            C, K = tensor.shape[0], tensor.shape[1]

            row = torch.arange(C).unsqueeze(-1).expand_as(tensor).to(tensor.device)
            col = torch.arange(K).unsqueeze(0).expand_as(tensor).to(tensor.device)

            output = torch.zeros((C, N, K), dtype=torch.float32).to(tensor.device)
            output[row, tensor, col] = 1.0

            return output

        # shifts.shape:            seg_shift^2 x 2
        # list_real.shape: C x K x              (2+1+eta)

        # Distance.shape:  C x K x seg_shift^2
        distance = torch.sum(torch.abs(shifts.unsqueeze(0).unsqueeze(1) + seg_size / 2 - (list_real[:, :, :2].unsqueeze(2) + padding)),
                             3)

        # argmin:          C x K
        topk = torch.argmin(distance, dim=2)

        #                  C x seg_shift^2 x K
        return k_hot_encoding(topk, distance.shape[2])

    def list_patch_loss(self, patch_pred, list_gt, seg_loc):
        patch_gt = torch.sum(self.find_closest(seg_loc, list_gt, self.seg_size, padding=self.n_pad), 2)
        return torch.sum(torch.square(patch_gt-patch_pred))/torch.sum(patch_gt)

    def list_cyc_loss(self, list_gt,list_pred,patch_pred,seg_loc):
        A         = self.get_cost_matrix_(list_gt,list_pred)
        num_data  = A.shape[0]
        num_dim   = A.shape[1]
        x_indices = np.zeros((num_data,num_dim),dtype=int)
        y_indices = np.zeros((num_data,num_dim),dtype=int)
        self.lst_elem_weight = self.lst_elem_weight.to(A.device)

        for i in range(num_data):
            An  = torch.sum(A[i,:,:,:3], -1)
            x_indices[i,:],y_indices[i,:] = lsa(An.detach().cpu().numpy())

        A         = torch.mean(self.lst_elem_weight*A, -1).view(num_data,-1)
        x_indices = torch.from_numpy(x_indices).to(A.device)
        y_indices = torch.from_numpy(y_indices).to(A.device)
        indices   = x_indices*num_dim + y_indices

        losses    = torch.mean(torch.gather(A,1,indices),1)

        loss      = torch.mean(losses) + self.lambda_loc*self.list_patch_loss(patch_pred, list_gt, seg_loc) # potential lambda

        return loss

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN-GP."""
        alpha = torch.rand(real_samples.shape[0], *[1] * (real_samples.ndim - 1), device=real_samples.device  )  
        interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        interpolated_output = D(interpolated)
        # Compute gradients w.r.t. interpolated samples
        gradients = autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Flatten gradients and compute penalty
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_gp*gradient_penalty

    def im_dis_loss(self,res_im_real,res_im_syn):
        return torch.mean(torch.square(res_im_real))+torch.mean(torch.square(1-res_im_syn))

    def list_dis_loss(self,res_list_real,res_list_syn):
        return torch.mean(torch.square(res_list_real))+torch.mean(torch.square(1-res_list_syn))

    def wasserstein_loss(self,real, syn, weights,dim=2):
        # Combine batch and elements into a single dimension for each feature
        real_reshaped = real.permute(*[i for i in range(real.ndimension()) if i != dim], dim).reshape(-1, real.size(dim)) 
        syn_reshaped = syn.permute(*[i for i in range(syn.ndimension()) if i != dim], dim).reshape(-1, syn.size(dim)) 

        # Sort both tensors along the combined sample dimension for each feature
        sorted_real, _ = torch.sort(real_reshaped, dim=0) 
        sorted_syn, _ = torch.sort(syn_reshaped, dim=0)   

        # Compute combined range (max and min across real and synthetic data)
        max_vals = torch.max(torch.max(real_reshaped, dim=0).values, torch.max(syn_reshaped, dim=0).values)  # Shape: (n_feature,)
        min_vals = torch.min(torch.min(real_reshaped, dim=0).values, torch.min(syn_reshaped, dim=0).values)  # Shape: (n_feature,)
        feature_ranges = max_vals - min_vals  # Shape: (n_feature,)
        feature_ranges = torch.clamp(feature_ranges, min=1e-8)  # Avoid division by zero
        loss = torch.sum(torch.abs(sorted_real - sorted_syn) / feature_ranges * weights,dim=-1) 
        return torch.mean(loss)

    def gen_loss(self,res_im_real,res_list_real,res_im_syn,res_list_syn,im_gt,im_pred,list_gt,patch_pred,seg_loc,list_pred, debug=False):
        cyc_loss_im = self.im_cyc_loss(im_gt,im_pred)
        cyc_loss_lst = self.list_cyc_loss(list_gt,list_pred,patch_pred,seg_loc)
        cyc_loss = self.lambda_cyc_img *cyc_loss_im + self.lambda_cyc_lst * cyc_loss_lst
        entropy_loss = -torch.mean(list_pred[...,2] * torch.log(list_pred[...,2] + 1e-8) + (1 - list_pred[...,2]) * torch.log(1 - list_pred[...,2] + 1e-8))
        regularization = self.lambda_entropy*entropy_loss
        tot_loss = self.lambda_dis_img*torch.mean(torch.square(res_im_syn)) + self.lambda_dis_lst*torch.mean(torch.square(res_list_syn)) + cyc_loss + regularization
        if debug:
            return tot_loss, cyc_loss_im, cyc_loss_lst
        return tot_loss
