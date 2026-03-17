

import numpy as np
import torch


class GeoRotaryEmbedding(torch.nn.Module):
    """Geospatial Rotary Positional Embedding

    based on https://github.com/limefax/rope-nd
    """
    def __init__(
        self, 
        emb_dim: int,
        num_coords: int, 
        limits: list[list[float]], 
        trunc_embed: float = 0., 
        gamma: float = 1., 
    ) -> GeoRotaryEmbedding:
        super().__init__()

        assert (emb_dim % (2*num_coords)) == 0, f'Embedding dim must divide by 2 * number of coordinates ({2*num_coords})'
        self.emb_dim = emb_dim // (2*num_coords)
        self.num_coords = num_coords
        assert len(limits) == num_coords, f'Length of limits much match num_coords ({num_coords})'
        self.limits = limits
        self.trunc_embed = trunc_embed
        self.gamma = gamma
        self.omegas = [
            self.get_omega(*limit) for limit in self.limits
        ]

    def get_omega(self, min_angle:float, max_angle:float) -> np.ndarray:
        """Get omega values to produce series of angles
        """
        trunc_emb_dim = int(self.emb_dim * self.trunc_embed)
        omega_pow = np.concat([np.ones(trunc_emb_dim), np.linspace(1, 0, self.emb_dim - trunc_emb_dim)])
        omega = 1 / (min_angle * (max_angle / min_angle)**(omega_pow**self.gamma))
        return omega

    def update(self, coords: torch.tensor) -> None:
        """Update angles with new set of coordinates

        should be performed at the start of each batch
        """
        B, N, L = coords.shape
        assert N == self.num_coords
        angles = torch.cat(
            [
                torch.einsum(
                    'bm,d->bmd', 
                    coords[:,i], 
                    torch.tensor(o, dtype=coords.dtype, device=coords.device)
                ) for i, o in enumerate(self.omegas)
            ], 
            -1
        )
        self.rotations = torch.polar(torch.ones_like(angles), angles)

    def forward(self, x: torch.tensor) -> torch.tensor:
        pe_x = self.rotations.unsqueeze(1) * torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        return torch.view_as_real(pe_x).reshape(x.shape)

def torch_circmean(x, min, max, *args):
    factor = (max - min)
    x = 2 * torch.pi * (x - min) / factor
    x = torch.atan2(
        torch.sin(x).mean(*args), 
        torch.cos(x).mean(*args), 
    )
    x = x * factor / (2 * torch.pi) + min
    return x