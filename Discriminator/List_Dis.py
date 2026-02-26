import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

import numpy as np
from einops import repeat

# helpers
def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes
class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask = None):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim = -1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest = False)

            v = batched_index_select(v, indices, dim = 2)
            qk_rel = batched_index_select(qk_rel, indices, dim = 2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim = 2)
            mask = batched_index_select(mask, indices, dim = 2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = torch.einsum('b i j d, b i j d -> b i d', attn, v)
        return agg

class ListDis_ANN(nn.Module):
    def __init__(self, data_dim, im_size, transformer_size=24, pos_mlp_hidden_dim=32, attn_mlp_hidden_mult=4, 
                 use_sn=True, dropout_prob=0.1, num_fc_layers=3, num_transformer_layers=3):
        super().__init__()

        self.eta_alpha_dim = data_dim + 1
        self.transformer_size = transformer_size
        self.image_size = im_size
        self.dropout_prob = dropout_prob
        self.num_fc_layers = num_fc_layers - 1
        self.num_transformer_layers = num_transformer_layers

        # Dropout applied after attention layers
        self.dropout = nn.Dropout(p=self.dropout_prob)



        # Transformer layers
        self.attn_layers = nn.ModuleList([
            PointTransformerLayer(
                dim=self.transformer_size,
                pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                attn_mlp_hidden_mult=attn_mlp_hidden_mult
            )
            for _ in range(self.num_transformer_layers)
        ])

        # First fully connected layer (always exists)
        self.fc1 = self._apply_sn(nn.Linear(self.eta_alpha_dim, self.transformer_size),use_sn)

        # Additional fully connected layers, including the last one
        self.fc_layers = nn.ModuleList([
            self._apply_sn(nn.Linear(self.transformer_size, self.transformer_size),use_sn)
            for _ in range(self.num_fc_layers - 1)
        ])
        self.fc_last = self._apply_sn(nn.Linear(self.transformer_size, 1),use_sn)  # Final layer

    @staticmethod
    def k_hot_encoding(tensor, N):
        # Transform from: C x K
        # to:             C x seg_shift^2 x K
        C, K = tensor.shape[0], tensor.shape[1]

        row = torch.arange(C).unsqueeze(-1).expand_as(tensor).to(tensor.device)
        col = torch.arange(K).unsqueeze(0).expand_as(tensor).to(tensor.device)

        output = torch.zeros((C, N, K), dtype=torch.float32).to(tensor.device)
        output[row, tensor, col] = 1.0

        return output

    @staticmethod
    def _apply_sn(layer, use_sn):
        """Applies proper weight initialization and Spectral Normalization if enabled."""
        # Perform orthogonal initialization first
        if hasattr(layer, 'weight') and layer.weight is not None:
            torch.nn.init.orthogonal_(layer.weight, gain=0.1)  # Or other initialization
        
        # Apply spectral normalization after initializing weights
        if use_sn:
            layer = nn.utils.spectral_norm(layer)
        
        return layer

    def forward(self, x):
        B = x.shape[0]
        xyz = torch.clone(x[:, :, :3])
        #xyz[xyz[:, :, 2]<0.5] = 0
        xyz[:, :, 2] = 0  # This is alpha but we make it z here

        mask = torch.zeros(x.shape[0], x.shape[1]).bool().to(x.device)
        feats = F.relu(self.fc1(x[...,2:].view(-1, self.eta_alpha_dim )))
        feats = self.dropout(feats)
        feats = feats.view(B, -1, self.transformer_size)

        # Pass through transformer layers
        for attn_layer in self.attn_layers:
            feats = attn_layer(feats, xyz  / self.image_size, mask=mask) + feats
            feats = self.dropout(feats)

        feats = feats.view(-1, self.transformer_size)

        # Pass through additional fully connected layers (if any)
        for fc_layer in self.fc_layers:
            feats = F.relu(fc_layer(feats))
            feats = self.dropout(feats)

        # Final output layer
        x = self.fc_last(feats)
        x = x.view(B, -1)
        x = torch.mean(x, 1)  # TODO

        return x

