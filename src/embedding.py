from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_2tuple

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2, 
        )

        self.identity = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.dropout(self.identity(x) + self.block(x))

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2, 
        )

        self.identity = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(in_ch, out_ch, 1)
        )
            
    
    def forward(self, x):
        return self.dropout(self.identity(x) + self.block(x))

class InputBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.conv1, self.norm1, self.relu1, self.conv2, 
        )

        self.identity = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.dropout(self.identity(x) + self.block(x))

class ResUNetEmbed(nn.Module):
    def __init__(self, input_channels: int, embed_dim: int, depth: int):
        super().__init__()
        self.channels = [embed_dim//2**(depth-i) for i in range(depth+1)]
        
        self.input_block = InputBlock(input_channels, self.channels[0])
        
        self.encode_blocks = nn.ModuleList(
            [ResBlock(self.channels[i], self.channels[i+1]) for i in range(depth)]
        )
        self.pool = nn.MaxPool2d(2)
        self.stride_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, embed_dim, kernel_size=2**(depth-i), stride=2**(depth-i)) 
                for i, in_ch in enumerate(self.channels)
            ]
        )
    
    def forward_features(self, x):
        features = []
        
        x = self.input_block(x)
        features.append(x)
        
        for block in self.encode_blocks:
            x = self.pool(x)
            x = block(x)
            features.append(x)
        return features
    
    def forward(self, x):
        features = self.forward_features(x)
        features = [conv(x) for conv, x in zip(self.stride_convs, features)]

        return torch.stack(features, dim=0).sum(0)
    
class ResUNetPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = 256,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = ResUNetEmbed(in_chans, embed_dim, int(np.log2(patch_size)))
        self.norm = norm_layer(embed_dim, **dd) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x