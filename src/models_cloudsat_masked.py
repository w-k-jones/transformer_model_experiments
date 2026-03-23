"""
A masked autoencoder model based on MAE and ViT-5 for CloudSat reconstruction
"""


from functools import partial

import torch
import torch.nn as nn

from .models_mae5 import GeoRoPEAttention, GeoMaskedAutoEncoder

class GeoCloudSatMaskedTransformer(GeoMaskedAutoEncoder):
    def __init__(self, *args, output_dim=125, prediction_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Decoder blocks with mask attention
        self.decoder_attn = partial(
            torch.nn.functional.scaled_dot_product_attention,
            attn_mask=None, 
        )
        self.Attention_block = partial(GeoRoPEAttention, attention_function=self.decoder_attn)
        dpr = [self.drop_path_rate for i in range(self.decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                self.block_layers(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    drop=0.0,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    Attention_block=self.Attention_block,
                    Mlp_block=self.Mlp_block,
                    rope=self.decoder_rope, 
                    num_registers=self.num_registers,
                    qk_norm=self.qk_norm,
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
    
    def update_attention_mask(self, coords, num_input_coords, num_output_coords, n_attn=32):
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

        self.decoder_attn.keywords["attn_mask"] = attn_mask.unsqueeze(1)

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
        self.update_coords(coords, self.decoder_rope)

        # update attention mask
        self.update_attention_mask(coords, num_input_coords, num_output_coords)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
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