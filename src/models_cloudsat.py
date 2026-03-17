"""
A masked autoencoder model based on MAE and ViT-5 for CloudSat reconstruction
"""


import torch
import torch.nn as nn

from .models_mae5 import GeoRoPEAttention, Block, GeoMaskedAutoEncoder

class GeoCloudSatTransformer(GeoMaskedAutoEncoder):
    def __init__(self, *args, output_dim=125, **kwargs):
        super().__init__(*args, **kwargs)
        
        # New decoder with translation features
        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         Block(
        #             dim=self.decoder_embed_dim,
        #             num_heads=self.decoder_num_heads,
        #             mlp_ratio=self.mlp_ratio,
        #             qkv_bias=self.qkv_bias,
        #             qk_scale=self.qk_scale,
        #             drop=0.0,
        #             attn_drop=self.attn_drop_rate,
        #             drop_path=self.drop_path_rate,
        #             norm_layer=self.norm_layer,
        #             act_layer=self.act_layer,
        #             Attention_block=GeoRoPEAttention,
        #             Mlp_block=self.Mlp_block,
        #             rope=self.decoder_rope, 
        #             num_registers=self.num_registers,
        #             qk_norm=self.qk_norm,
        #             layer_scale=self.layer_scale,
        #         )
        #         for i in range(self.decoder_depth)
        #     ]
        # )
        self.output_dim = output_dim
        self.output_head = nn.Linear(self.decoder_embed_dim, self.output_dim)

    def unshuffle_tokens(self, x, ids_restore):
        end = -self.num_registers if self.num_registers else None
        x_ = torch.gather(x[:, 1:end], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.num_registers:
            x = torch.cat([x[:, :1], x_, x[:, -self.num_registers:]], dim=1)  # append cls and register tokens
        else:
            x = torch.cat([x[:, :1], x_], dim=1)  # append cls token

        return x

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
        rel_output_coords = output_coords - mean_coords.unsqueeze(-1)
        coords = torch.cat([coords, rel_output_coords], dim=-1)
        self.update_coords(coords, self.decoder_rope)

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