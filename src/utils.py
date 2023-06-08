from dataclasses import dataclass, field
import torch
from typing import Literal, NamedTuple
import numpy as np


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    data_path: str = "tiny_nerf_data.npz"
    near_thresh: float = 2.0
    far_thresh: float = 6.0
    num_samples: int = 32
    lr: float = 5e-3
    num_iters: int = 1000
    seed: int = 9458
    filter_size: int = 256
    L_origin: int = 10
    L_direction: int = 4
    train_test_split_ratio: float = 0.9
    dtype: torch.dtype = torch.float32
    device: torch.device = field(
        default_factory=lambda: (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )


@dataclass
class NerfData:
    img_train: torch.Tensor
    pose_train: torch.Tensor
    focal_length: torch.Tensor
    img_test: torch.Tensor
    pose_test: torch.Tensor
    height: int
    width: int
    device: torch.device

    @property
    def H(self):
        return self.height

    @property
    def W(self):
        return self.width

    def sample_trainset(self):
        """sample image and pose matrix"""
        idx = np.random.randint(self.img_train.shape[0])

        target_img = self.img_train[idx].to(self.device)
        target_pose = self.pose_train[idx].to(self.device)
        return (target_img,
                target_pose)

    def sample_testset(self):
        """sample image and pose matrix"""
        idx = np.random.randint(self.img_test.shape[0])

        target_img = self.img_test[idx].to(self.device)
        target_pose = self.pose_test[idx].to(self.device)
        return (target_img,
                target_pose)


def load_nerf_dataset(config: Config) -> NerfData:
    data = np.load(config.data_path)
    N: int = data["images"].shape[0]
    num_train = int(config.train_test_split_ratio * N)
    indice = np.arange(N)
    np.random.shuffle(indice)
    train_idx = indice[:num_train]
    test_idx = indice[num_train:]

    pose_train = torch.from_numpy(   # Camera extrinsics (poses)
        data["poses"][train_idx]
    )
    pose_test = torch.from_numpy(          # Camera extrinsics (poses)
        data["poses"][test_idx]
    )

    focal_length = torch.from_numpy(      # Focal length (intrinsics)
        data["focal"]
    )

    img_train = torch.from_numpy(
        data["images"][train_idx]
    )
    img_test = torch.from_numpy(
        data["images"][test_idx]
    )

    _, H, W, _ = img_train.shape

    return NerfData(img_train,
                    pose_train,
                    focal_length,
                    img_test,
                    pose_test,
                    H,
                    W,
                    config.device)


def positional_encoder(tensor: torch.Tensor,
                       L: int) -> torch.Tensor:
    encoding = [tensor]
    frequency_bands = 2.0 ** torch.arange(
        0,
        L,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    for freq in frequency_bands:
        encoding.append(torch.sin(tensor * freq))
        encoding.append(torch.cos(tensor * freq))

    return torch.cat(encoding, dim=-1)
