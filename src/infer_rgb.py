import torch.nn.functional as F
from typing import Tuple, NamedTuple
from .utils import Config
from typing import Callable, Optional
import torch
from .get_rays import get_rays, Rays
from .get_query_points import get_query_points, get_t_vals, QueryPoints
from .Nerf import Nerf
from .render_rgb_depth import render_rgb_depth, RgbInferenceResult


def infer_rgb(pose: torch.Tensor,
              nerf_model: Nerf,
              H: int,
              W: int,
              focal_length: float,
              near_thresh: float,
              far_thresh: float,
              num_samples: int,
              dtype: torch.dtype,
              device: torch.device
              ) -> RgbInferenceResult:
    assert pose.shape == (4, 4)
    ray: Rays = get_rays(H,
                         W,
                         focal_length,
                         pose,
                         dtype,
                         device)

    Q: QueryPoints = get_query_points(
        ray.origins,
        ray.directions,
        get_t_vals(H, W, near_thresh,
                   far_thresh, num_samples,
                   device)
    )
    rgbs, sigmas = nerf_model.forward(
        Q.r_vals,
        Q.d_vals
    )

    return render_rgb_depth(
        rgbs,
        sigmas,
        Q.t_vals,
        dtype,
        device,
    )
