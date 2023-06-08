from typing import NamedTuple
from .utils import Config
import torch


class Rays(NamedTuple):
    origins: torch.Tensor
    directions: torch.Tensor


def get_rays(H: int,
             W: int,
             focal_length: float,
             pose: torch.Tensor,
             dtype: torch.dtype,
             device: torch.device,
             ) -> Rays:
    camera_matrix = pose[:3, :3]
    ray_origins = pose[:3, -1]
    assert ray_origins.shape == (3,)
    ray_origins = ray_origins.expand(H, W, 3)

    (ii, jj) = torch.meshgrid(
        torch.arange(W,
                     dtype=dtype,
                     device=device),
        torch.arange(H,
                     dtype=dtype,
                     device=device),
        indexing="xy",
    )
    assert ii.shape == jj.shape == (H,
                                    W)
    ii = (ii - W * .5) / focal_length
    jj = (jj - H * .5) / focal_length

    directions = torch.stack([ii,
                              -jj,
                              -torch.ones_like(ii)],
                             dim=-1)
    assert directions.shape == (H, W, 3)

    ray_directions = directions @ camera_matrix.transpose(1, 0)

    return Rays(ray_origins,
                ray_directions)
