from .utils import Config, positional_encoder
import torch
import torch.nn as nn
from typing import Tuple


class Nerf(torch.nn.Module):

    def __init__(self, config: Config):

        super(Nerf, self).__init__()
        self.config = config
        origin_enc_dim: int = (
            3 + 3 * 2 * config.L_origin
        )
        dir_enc_dim: int = (
            3 + 3 * 2 * config.L_direction
        )

        self.layer1 = nn.Sequential(
            nn.Linear(
                origin_enc_dim,
                config.filter_size
            ),
            nn.ReLU(),
            nn.Linear(
                config.filter_size,
                config.filter_size
            ),
            nn.ReLU(),
            nn.Linear(
                config.filter_size,
                config.filter_size
            ),
            nn.ReLU(),
            nn.Linear(
                config.filter_size,
                config.filter_size
            ),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(origin_enc_dim + config.filter_size,
                      config.filter_size),
            nn.ReLU(),
            nn.Linear(config.filter_size,
                      config.filter_size),
            nn.ReLU(),
            nn.Linear(config.filter_size,
                      config.filter_size),
            nn.ReLU(),
            nn.Linear(config.filter_size,
                      config.filter_size),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(config.filter_size + dir_enc_dim,
                      config.filter_size),
            nn.ReLU(),
            nn.Linear(config.filter_size,
                      config.filter_size//2),
            nn.ReLU(),
            nn.Linear(config.filter_size//2,
                      4),
        )
        self.to(device=config.device)

    def forward(self,
                r_vals: torch.Tensor,
                d_vals: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        (H, W, num_samples, C) = r_vals.size()
        assert r_vals.size() == d_vals.size() and C == 3
        r_vals = r_vals.reshape(-1, 3)
        d_vals = d_vals.reshape(-1, 3)

        L_o = self.config.L_origin
        L_d = self.config.L_direction

        r_vals = positional_encoder(r_vals, L_o)
        d_vals = positional_encoder(d_vals, L_d)

        h = self.layer1(r_vals)
        h = self.layer2(torch.cat([h, r_vals],
                                  dim=-1))
        pred = self.layer3(torch.cat([h, d_vals],
                                     dim=-1))
        pred = pred.reshape(H, W, num_samples, 4)
        rgb = pred[..., :3]
        sigma = pred[..., 3]
        rgb = torch.sigmoid(rgb)
        sigma = nn.functional.relu(sigma)
        return (rgb, sigma)
