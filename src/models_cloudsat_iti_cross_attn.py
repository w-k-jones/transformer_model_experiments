"""
A masked autoencoder model based on MAE and ViT-5 from Geostation to CloudSat translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

from .models_cloudsat import GeoCloudSatTransformer
from .models_mae5 import  GeoRoPEAttention

class GeoITICrossAttention(nn.Module):
    """Geospatial aware image-to-image attention
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rope=None,
        latent_rope=None,
        num_registers=0,
        qk_norm=False,
        attention_function=F.scaled_dot_product_attention, 
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_latent = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_latent = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_registers = num_registers
        self.rope = rope or nn.Identity()
        self.latent_rope = latent_rope or nn.Identity()
        self.attention_function = attention_function

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.q_norm_latent = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm_latent = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, latent):
        B, N = x.shape[:2]
        B, L = latent.shape[:2]

        # Prediction q,k,v
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # shape 2,B,heads,N,head_dim
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

        # Latent q,k,v
        qkv_l = self.qkv_latent(latent)
        qkv_l = qkv_l.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # shape 2,B,heads,N,head_dim
        q_l, k_l, v_l = qkv_l.unbind(0)  # B, num_heads, N, head_dim

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q_l = self.q_norm_latent(q_l)
            k_l = self.k_norm_latent(k_l)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        if self.latent_rope is not None:
            q_l = self.latent_rope(q_l)
            k_l = self.latent_rope(k_l)

        x = self.attention_function(q, k_l, v_l)
        x = self.attn_drop(x)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        latent = self.attention_function(q_l, k, v)
        latent = self.attn_drop(latent)
        latent = latent.transpose(1, 2).reshape(B, L, -1)
        latent = self.proj_latent(latent)
        latent = self.proj_drop(latent)

        return x, latent


class ITICrossBlock(nn.Module):
    """Translation block: first performs self attention, then attention against keys/values from latent space representation
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=torch.nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        init_values=1e-4,
        rope=None,
        latent_rope=None,
        num_registers=0,
        qk_norm=False,
        layer_scale=True,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm1 = norm_layer(dim)
        self.latent_norm1 = norm_layer(dim)
        self.attn = GeoITICrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope=rope,
            latent_rope=latent_rope, 
            num_registers=num_registers,
            qk_norm=qk_norm,
        )

        self.norm2 = norm_layer(dim)
        self.latent_norm2 = norm_layer(dim)
        self.ffd = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ffd_latent = Mlp_block(
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
            self.gamma_latent_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_latent_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x, latent):
        if self.layer_scale:
            x_, latent_ = self.attn(self.norm1(x), self.latent_norm1(latent))
            x = x + self.drop_path(self.gamma_1 * x_)
            x = x + self.drop_path(self.gamma_2 * self.ffd(self.norm2(x)))
            latent = latent + self.drop_path(self.gamma_latent_1 * latent_)
            latent = latent + self.drop_path(self.gamma_latent_2 * self.ffd_latent(self.latent_norm2(latent)))
        else:
            x_, latent_ = self.attn(self.norm1(x), self.latent_norm1(latent))
            x = x + self.drop_path(x_)
            x = x + self.drop_path(self.ffd(self.norm2(x)))
            latent = latent + self.drop_path(latent_)
            latent = latent + self.drop_path(self.ffd_latent(self.latent_norm2(latent)))
        return x, latent


class GeoCloudSatITICrossTransformer(GeoCloudSatTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # New decoder with translation features
        self.decoder_blocks = nn.ModuleList(
            [
                ITICrossBlock(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    drop=0.0,
                    attn_drop=self.attn_drop_rate,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer, 
                    Mlp_block=self.Mlp_block,
                    rope=self.decoder_rope, 
                    latent_rope=self.encoder_rope, 
                    num_registers=self.num_registers,
                    qk_norm=self.qk_norm,
                    layer_scale=self.layer_scale,
                )
                for i in range(self.decoder_depth)
            ]
        )

    def forward_decoder(self, latent, input_coords, output_coords):
        # embed tokens
        latent = self.decoder_embed(latent)
        B = latent.shape[0]

        # create output from mask tokens sequence
        x = self.mask_token.repeat(B, output_coords.shape[-1], 1)
        # Append class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # embed coords
        mean_coords = self.mean_coords(input_coords.reshape(*input_coords.shape[:2], -1))
        rel_output_coords = output_coords - mean_coords.unsqueeze(-1)
        self.update_coords(rel_output_coords, self.decoder_rope, class_token=True) # don't need to update encoder coords

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, latent = blk(x, latent)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.output_head(x[:,1:]) # don't return class token

        return x

    def forward(self, imgs, input_coords, output_coords):
        latent, _, _ = self.forward_encoder(imgs, input_coords, 0.)
        pred = self.forward_decoder(latent, input_coords, output_coords)  # [N, L, p*p*C]
        return pred