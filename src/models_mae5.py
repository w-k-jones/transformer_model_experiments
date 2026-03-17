"""
A masked autoencoder model based on MAE and ViT-5 with geopositional encoding
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed

from timm.models.layers import DropPath, trunc_normal_

from .georope import GeoRotaryEmbedding
from .utils import torch_circmean


class GeoRoPEAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rope=None,
        num_registers=0,
        qk_norm=False,
        attention_function=F.scaled_dot_product_attention, 
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_registers = num_registers
        self.rope = rope
        self.attention_function = attention_function

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # shape 3,B,heads,N,head_dim
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        x = self.attention_function(q, k, v)
        x = self.attn_drop(x)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
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
        act_layer=torch.nn.SiLU,
        norm_layer=nn.LayerNorm,
        Attention_block=GeoRoPEAttention,
        Mlp_block=Mlp,
        init_values=1e-4,
        rope=None,
        num_registers=0,
        qk_norm=False,
        layer_scale=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope=rope,
            num_registers=num_registers,
            qk_norm=qk_norm,
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

    def forward(self, x):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GeoMaskedAutoEncoder(nn.Module):
    """GeoMAE based on ViT-5 with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        block_layers: nn.Module = Block,
        Patch_layer: nn.Module = PatchEmbed,
        act_layer: nn.Module = nn.GELU,
        Attention_block: nn.Module = GeoRoPEAttention,
        Mlp_block: nn.Module = Mlp,
        init_scale: float = 1e-4,
        use_registers: bool = False,
        qk_norm: bool = True,
        layer_scale: bool = True,
        num_coords: int = 4,
        coord_limits: list[list[float]] = [
            [1  / (2*np.pi), 360 / (2*np.pi)], # longitude 1->360 degrees
            [1  / (2*np.pi), 360 / (2*np.pi)], # latitude 1->360 degrees
            [(1/(24*60)) / (2*np.pi), 1 / (2*np.pi)], # time of day 1 minute -> 1 day
            [1, 10000], # epoch 2 pi days -> 20000 pi days (~80 years)
        ],
        rope_trunc: float = 0.25, 
        **kwargs
    ):
        super().__init__()
        self.dropout_rate = drop_rate
        self.num_features = self.embed_dim = embed_dim
        img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.block_layers = block_layers
        self.Patch_layer = Patch_layer
        self.act_layer = act_layer
        self.Attention_block = Attention_block
        self.Mlp_block = Mlp_block
        self.init_scale = init_scale
        self.use_registers = use_registers
        self.qk_norm = qk_norm
        self.layer_scale = layer_scale
        self.num_coords = num_coords
        self.coord_limits = coord_limits
        self.rope_trunc = rope_trunc
        

        # Image embedding - todo: enable conv stem option
        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.num_coords = num_coords
        self.num_registers = int(2**num_coords) if use_registers else 0
        self.coord_limits = coord_limits
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = (
            nn.Parameter(torch.zeros(1, self.num_registers, embed_dim))
            if self.num_registers > 0
            else None
        )
        self.encoder_rope = GeoRotaryEmbedding(
            embed_dim//num_heads, num_coords, coord_limits, rope_trunc, 
        )

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=GeoRoPEAttention,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                    rope=self.encoder_rope,
                    num_registers=self.num_registers,
                    qk_norm=qk_norm,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_rope = GeoRotaryEmbedding(
            decoder_embed_dim//decoder_num_heads, num_coords, coord_limits, rope_trunc, 
        )

        dpr = [drop_path_rate for i in range(decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                block_layers(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=GeoRoPEAttention,
                    Mlp_block=Mlp_block,
                    rope=self.decoder_rope, 
                    num_registers=self.num_registers,
                    qk_norm=qk_norm,
                    layer_scale=layer_scale,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        if self.num_registers > 0:
            trunc_normal_(self.reg_token, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "reg_token"}
    

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)

        todo: remove fixed length channels dim
        """
        p = self.patch_embed.patch_size[0]
        N, C, H, W = imgs.shape
        assert H == W and H % p == 0

        h = w = H // p
        x = imgs.reshape(shape=(N, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(N, h * w, p**2 * C))
        return x


    def mean_coords(self, coords):
        mean_coords = torch.stack(
            [
                torch_circmean(coords[:, i], 0, self.coord_limits[i][1], -1) for i in range(self.num_coords)
            ], 1
        )
        return mean_coords # shape N, C
    

    def patchify_coords(self, coords, mean_coords=None):
        """
        imgs: (N, C, H, W)
        x: (N, L, C*patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        N, C, H, W = coords.shape
        assert H == W and H % p == 0

        h = w = H // p
        x = coords.reshape(shape=(N, C, h, p, w, p))
        x = torch.einsum('NChpwq->NChwpq', x)
        x = x.reshape(shape=(N, C, h * w, p**2))
        if mean_coords is not None:
            # Normalise around centre coords
            x = x - mean_coords.reshape(N, C, 1, 1)
        return self.mean_coords(x)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)

        todo: remove fixed length channels dim
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_keep, ids_restore
    

    def update_coords(self, coords, rope, class_token=True):
        B, N, _ = coords.shape

        # normalise coords
        coords = coords - self.mean_coords(coords).unsqueeze(-1)

        # Append class token and register tokens
        if self.num_registers > 0:
            minmax_coords = torch.stack(
                [coords.min(-1)[0], coords.max(-1)[0]], -1
            )
            
            register_coords = torch.stack(
                [torch.cartesian_prod(*minmax_coords[i]).transpose(1,0) for i in range(B)], 
                0
            )
            coords = torch.cat(
                [coords, register_coords], -1
            )
        if class_token:
            coords = torch.cat(
                [torch.zeros(*coords.shape[:-1], 1, dtype=coords.dtype, device=coords.device), coords], -1
            )

        # update RoPE angles
        rope.update(coords)


    def forward_encoder(self, x, coords, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_keep, ids_restore = self.random_masking(x, mask_ratio)

        coords = self.patchify_coords(coords)
        coords_masked = torch.gather(coords, dim=2, index=ids_keep.unsqueeze(1).repeat(1, self.num_coords, 1))
        self.update_coords(coords_masked, self.encoder_rope)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # append register tokens
        if self.num_registers:
            reg_tokens = self.reg_token.repeat(x.shape[0], 1, 1)
            x = torch.cat((x, reg_tokens), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    

    def forward_decoder(self, x, coords, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 + self.num_registers - x.shape[1], 1)
        end = -self.num_registers if self.num_registers else None
        x_ = torch.cat([x[:, 1:end], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if self.num_registers:
            x = torch.cat([x[:, :1], x_, x[:, -self.num_registers:]], dim=1)  # append cls and register tokens
        else:
            x = torch.cat([x[:, :1], x_], dim=1)  # append cls token

        # embed coords
        coords = self.patchify_coords(coords)
        self.update_coords(coords, self.decoder_rope)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token and register tokens
        x = x[:, 1:end]

        return x
    

    def forward(self, imgs, coords, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, coords, mask_ratio)
        pred = self.forward_decoder(latent, coords, ids_restore)  # [N, L, p*p*C]
        return pred, mask


    def forward_loss(self, imgs, coords, mask_ratio=0.75):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        pred, mask = self.forward(imgs, coords, mask_ratio=mask_ratio)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, pred, mask
