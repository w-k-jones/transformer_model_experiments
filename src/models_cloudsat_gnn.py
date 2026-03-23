"""
A masked autoencoder model based on MAE and ViT-5 for CloudSat reconstruction
"""


from functools import partial

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse

from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from .models_mae5 import GeoMaskedAutoEncoder

class GATBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=torch.nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        init_values=1e-4,
        layer_scale=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = gnn.GATv2Conv(
            dim,
            dim // num_heads, 
            heads=num_heads, 
            add_self_loops=False, 
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x, edge_index):
        B, L, E = x.shape

        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).reshape(-1, E), edge_index).reshape(B, L, E))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x).reshape(-1, E), edge_index).reshape(B, L, E))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GeoCloudSatMaskedGNN(GeoMaskedAutoEncoder):
    def __init__(self, *args, output_dim=125, prediction_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Decoder blocks with mask attention
        self.decoder_attn = partial(
            torch.nn.functional.scaled_dot_product_attention,
            attn_mask=None, 
        )
        dpr = [self.drop_path_rate for i in range(self.decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [   
                GATBlock(
                    self.decoder_embed_dim, 
                    self.decoder_num_heads, 
                    mlp_ratio=self.mlp_ratio,
                    drop=0.0,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    Mlp_block=self.Mlp_block,
                    layer_scale=self.layer_scale,
                )
                for i in range(self.decoder_depth)
            ]
        )
        self.output_dim = output_dim
        self.output_head = prediction_head or nn.Linear(self.decoder_embed_dim, self.output_dim)

    def unshuffle_tokens(self, x, ids_restore):
        end = -self.num_registers if self.num_registers else None
        x_ = torch.gather(x[:, 1:end], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.num_registers:
            x = torch.cat([x[:, :1], x_, x[:, -self.num_registers:]], dim=1)  # append cls and register tokens
        else:
            x = torch.cat([x[:, :1], x_], dim=1)  # append cls token

        return x
    
    def get_edge_index(self, coords, num_input_coords, num_output_coords, n_attn=8):
        B, N, L = coords.shape
        coords_dist = ((coords.unsqueeze(-1)-coords.unsqueeze(-2))**2).sum(1)
        attn_mask = torch.zeros(B, L+1, L+1, dtype=torch.bool, device=coords.device)

        # Set class tokens to True for all
        attn_mask[...,0,:] = True
        attn_mask[...,:,0] = True

        # Calc nearest n_attn input coord locations
        _, idx_input_mask = torch.sort(coords_dist[...,:num_input_coords], dim=-1)
        max_dist = torch.take_along_dim(
            coords_dist[...,:num_input_coords], idx_input_mask[...,n_attn-1:n_attn], dim=-1
        )
        attn_mask[...,1:,1:1+num_input_coords] = coords_dist[...,:num_input_coords] <= max_dist

        # Calc nearest n_attn output coord locations
        _, idx_output_mask = torch.sort(
            coords_dist[...,num_input_coords:num_input_coords+num_output_coords], dim=-1
        )
        max_dist = torch.take_along_dim(
            coords_dist[...,num_input_coords:num_input_coords+num_output_coords], 
            idx_output_mask[...,n_attn-1:n_attn], 
            dim=-1, 
        )
        attn_mask[...,1:,1+num_input_coords:1+num_input_coords+num_output_coords] = (
            coords_dist[...,num_input_coords:num_input_coords+num_output_coords] <= max_dist
        )

        return dense_to_sparse(attn_mask)[0]

    def forward_decoder(self, latent, input_coords, output_coords):
        # embed tokens
        latent = self.decoder_embed(latent)
        B, L, E = latent.shape

        # create output from mask tokens sequence
        x = torch.cat([
            latent, self.mask_token.repeat(B, output_coords.shape[-1], 1)
        ], dim=1)

        # embed coords
        mean_coords = self.mean_coords(input_coords.reshape(*input_coords.shape[:2], -1))
        coords = self.patchify_coords(input_coords, mean_coords=mean_coords)
        num_input_coords = coords.shape[-1]
        rel_output_coords = output_coords - mean_coords.unsqueeze(-1)
        num_output_coords = rel_output_coords.shape[-1]
        coords = torch.cat([coords, rel_output_coords], dim=-1)
        # self.update_coords(coords, self.decoder_rope)

        # update attention mask
        edge_index = self.get_edge_index(coords, num_input_coords, num_output_coords)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, edge_index)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.output_head(x[:, L:])

        return x
    

    def forward(self, imgs, input_coords, output_coords):
        latent, _, ids_restore = self.forward_encoder(imgs, input_coords, 0.)
        latent = self.unshuffle_tokens(latent, ids_restore)
        pred = self.forward_decoder(latent, input_coords, output_coords)  # [N, L, p*p*C]
        return pred
    
    
    def forward_loss(self, imgs, input_coords, target, output_coords):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        pred = self.forward(imgs, input_coords, output_coords)
        loss = (pred - target) ** 2
        return loss.mean()
    
def torch_circmean(x, min, max, *args, **kwargs):
    factor = (max - min)
    x = 2 * torch.pi * (x - min) / factor
    x = torch.atan2(
        torch.sin(x).mean(*args, **kwargs), 
        torch.cos(x).mean(*args, **kwargs), 
    )
    x = x * factor / (2 * torch.pi) + min
    return x