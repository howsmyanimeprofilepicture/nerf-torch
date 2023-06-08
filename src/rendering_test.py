import torch
from .utils import Config, NerfData
from .Nerf import Nerf
from .infer_rgb import infer_rgb
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def rendering_test(config: Config,
                   data: NerfData,
                   nerf_model: Nerf,
                   ):
    t_vals = np.linspace(config.far_thresh,
                         config.near_thresh,
                         config.num_samples)

    (tgt_img,
     tgt_pose) = data.sample_trainset()
    rgb_pred, weights = infer_rgb(
        tgt_pose,
        nerf_model=nerf_model,
        H=data.H,
        W=data.W,
        focal_length=data.focal_length,
        near_thresh=config.near_thresh,
        far_thresh=config.far_thresh,
        num_samples=config.num_samples,
        dtype=config.dtype,
        device=config.device
    )
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3,
                                      figsize=(8, 3))
    ax1.imshow(rgb_pred.detach().cpu())
    weights = weights.detach().cpu()
    ax2.imshow((weights * t_vals).sum(-1))
    ax3.imshow(tgt_img.detach().cpu())
    plt.show()
