import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class ImDis_ANN(nn.Module):
    def __init__(self, im_dim, im_size, num_channels=16, num_inter_layers=3, use_norm = False, use_sn=True, dropout_prob=0.1):
        super(ImDis_ANN, self).__init__()
        
        self.im_dim = im_dim
        self.im_size = im_size
        self.num_channels = num_channels
        self.num_inter_layers = num_inter_layers

        layers = []

        layers.append(self._apply_sn(nn.Conv2d(im_dim, num_channels, kernel_size=3, padding=1), use_sn))
        if use_norm:
            layers.append(nn.GroupNorm(4, num_channels))
        layers.append(nn.Dropout2d(p=dropout_prob))

        # Intermediate layers
        current_channels = num_channels
        for _ in range(num_inter_layers):
            next_channels = current_channels * 2
            layers.append(self._apply_sn(nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1), use_sn))
            if use_norm:
                layers.append(nn.GroupNorm(4, next_channels))
            layers.append(nn.Dropout2d(p=dropout_prob))  # Dropout after each intermediate layer
            layers.append(nn.LeakyReLU())
            layers.append(nn.MaxPool2d(2, stride=2, return_indices=False))
            current_channels = next_channels

        # Final convolution to 1 channel
        layers.append(self._apply_sn(nn.Conv2d(current_channels, 1, kernel_size=3, padding=1), use_sn))

        self.conv_layers = nn.ModuleList(layers)

    @staticmethod
    def _apply_sn(layer, use_sn):
        """Applies proper weight initialization and Spectral Normalization if enabled."""
        # Perform orthogonal initialization first
        if hasattr(layer, 'weight') and layer.weight is not None:
            torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Apply spectral normalization after initializing weights
        if use_sn:
            layer = nn.utils.spectral_norm(layer)
        
        return layer

    def forward(self, x):
        for layer in self.conv_layers[:-1]:  # Apply all but the last two layers with pooling, activation, and dropout
            x = layer(x)

        # Apply the final convolution layer (without pooling and dropout)
        x = self.conv_layers[-1](x)

        # Global mean pooling to produce the final output
        x = torch.mean(x, dim=(1, 2, 3))  # Shape: (N)
        return x
