# @title get_query.py

from typing import Any, Tuple, NamedTuple
import torch
from .utils import Config


class QueryPoints(NamedTuple):
    r_vals: torch.Tensor
    d_vals: torch.Tensor
    t_vals: torch.Tensor


def get_t_vals(H: int,
               W: int,
               near_thresh: float,
               far_thresh: float,
               num_samples: int,
               device: str):
    t_vals = torch.linspace(near_thresh,
                            far_thresh,
                            num_samples,
                            device=device)

    noise_shape = (H, W, num_samples,)
    t_vals = (
        t_vals
        + torch.rand(noise_shape).to(device=device)
        * (far_thresh - near_thresh)
        / num_samples
    )
    assert t_vals.shape == noise_shape
    return t_vals


def get_query_points(ray_origins: torch.Tensor,
                     ray_directions: torch.Tensor,
                     t_vals: torch.Tensor,
                     ) -> QueryPoints:
    (H, W, num_samples) = t_vals.shape
    (H, W, C) = ray_origins.shape
    assert ray_origins.shape == ray_directions.shape

    r_vals = (
        # (width, height, 1, 3)
        ray_origins[..., None, :] +
        # (width, height, 1, 3)
        ray_directions[..., None, :]
        # (width, height, depth_samples_per_ray, 1)
        * t_vals[..., :, None]
    )
    assert r_vals.shape == (H, W, num_samples, 3)
    d_vals = ray_directions[..., None, :].expand(
        r_vals.shape
    )
    return QueryPoints(r_vals,
                       d_vals,
                       t_vals)
