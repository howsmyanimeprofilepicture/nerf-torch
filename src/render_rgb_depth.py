import torch.nn.functional as F
from typing import Any, Tuple, NamedTuple
from .utils import Config
from typing import Callable, Optional
import torch


def _cumprod(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, dim=-1)
    # https://pytorch.org/docs/stable/generated/torch.roll.html
    cumprod = torch.roll(cumprod, 1, dims=-1)
    cumprod[..., 0] = 1.
    return cumprod


class RgbInferenceResult(NamedTuple):
    rgb: torch.Tensor
    depth_weights: torch.Tensor


def render_rgb_depth(rgbs: torch.Tensor,
                     sigmas: torch.Tensor,
                     t_vals: torch.Tensor,
                     dtype: torch.dtype,
                     device: torch.device,) -> RgbInferenceResult:

    (H, W, num_samples, C) = rgbs.size()
    assert t_vals.size() == sigmas.size() == (H, W, num_samples)
    one_e_10 = torch.tensor(
        [1e10],
        dtype=dtype,
        device=device
    )
    one_e_10 = one_e_10.expand(t_vals[..., :1].shape)
    deltas = torch.cat([
        t_vals[..., 1:] - t_vals[..., :-1],
        one_e_10
    ],
        dim=-1
    )

    alphas = 1. - torch.exp(-sigmas * deltas)
    transmittance = _cumprod(1. - alphas + 1e-10)
    weights = alphas * transmittance
    assert (weights.size()
            == deltas.size()
            == alphas.size()
            == (H, W, num_samples))
    rgb_pred = (weights[..., None] * rgbs).sum(dim=-2)

    return RgbInferenceResult(
        rgb_pred,
        weights
    )
