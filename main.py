from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass, field

from src import (Config,
                 load_nerf_dataset,
                 NerfData,
                 set_seed,
                 infer_rgb,
                 Nerf,
                 rendering_test)


def main():
    config = Config(
        data_path="tiny_nerf_data.npz",
        near_thresh=2.0,
        far_thresh=6.0,
        num_samples=32,
        lr=0.0001,
        num_iters=1000,
        seed=42,
        filter_size=256,
        L_origin=10,
        L_direction=4,
        train_test_split_ratio=0.9,
        dtype=torch.float32,
        device=(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        ),
    )
    set_seed(config.seed)
    data: NerfData = load_nerf_dataset(config)

    # plt.imshow(data.test_image.detach().cpu().numpy())
    # plt.show()

    display_every = 100
    nerf_model = Nerf(config)

    optimizer = torch.optim.Adam(
        nerf_model.parameters(),
        lr=config.lr
    )

    infer_kwargs = dict(
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

    loss_history = []
    cum_loss = 0.
    for i in range(config.num_iters):
        (tgt_img,
         tgt_pose) = data.sample_trainset()
        rgb_predicted, _ = infer_rgb(tgt_pose,
                                     **infer_kwargs)
        print(rgb_predicted.sum())
        loss = F.mse_loss(
            rgb_predicted,
            tgt_img,
        )
        cum_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % display_every == display_every-1:
            print(cum_loss/display_every)
            loss_history.append(cum_loss/display_every)
            cum_loss = 0.
            rendering_test(config, data, nerf_model)

    print('Done!')


if __name__ == "__main__":
    main()
