"""
datasets.py

Given a set of demonstrations, create a torch.Dataset with all the bells & whistles for training. Supports learning
from both RGB data and (WIP) joint states.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class RGBDataset(Dataset):
    def __init__(self, demonstrations: List[Path], action_space: str) -> None:
        """Flatten out each demonstration into the requisite set of state & action pairs."""
        self.demonstrations, self.action_space = demonstrations, action_space
        self.rgbs, self.proprios, self.x_actions, self.proprioceptive_dim, self.x_action_dim = [], [], [], None, None

        # Actual "actions" are a concatenation of ee and gripper actions...
        self.actions, self.action_dim, self.action_mean, self.action_std = None, None, None, None

        # ImageNet Default Transform
        self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # Get Samples
        self.get_samples()

    def get_samples(self) -> None:
        for d in self.demonstrations:
            # Data --> By default has the following keys:
            #   > q :: [traj_len - 1, 7]
            #   > qdot :: [traj_len - 1, 7]
            #   > delta_q: [traj_len - 1, 7] --> Note, shifted one to the right!
            #   > ee_pose :: [traj_len - 1, 7] (3-DoF position, 4-DoF quaternion for pose)
            #   > delta_ee_pose :: [traj_len - 1, 7] --> Note, shifted one to the right!
            #   > rgb :: [traj_len, 120, 160, 3]
            #   > gripper_open :: [traj_len, bool]
            data = np.load(d)

            # Switch on `action_space`
            if self.action_space == "endeff-delta":
                # =>> Note -- "delta" fields are offset by 1 timestep to the right!
                self.rgbs.extend(data["rgb"][:-1])
                self.proprios.extend(np.concatenate([data["q"][:-1], data["ee_pose"][:-1]], axis=1))
                self.x_actions.extend(data["delta_ee_pose"][1:])

            elif self.action_space == "endeff-delta-window":
                # Use a sliding window of size 10 to predict poses...
                idxs = sliding_window_view(np.arange(len(data["rgb"])), window_shape=10)[:, (0, -1)]
                self.rgbs.extend(data["rgb"][idxs[:, 0]])
                self.proprios.extend(np.concatenate([data["q"][idxs[:, 0]], data["ee_pose"][idxs[:, 0]]], axis=1))
                self.x_actions.extend(data["ee_pose"][idxs[:, 0]] - data["ee_pose"][idxs[:, 1]])

            elif self.action_space == "endeff":
                # =>> Note -- "delta" fields are offset by 1 timestep to the right!
                self.rgbs.extend(data["rgb"][:-1])
                self.proprios.extend(np.concatenate([data["q"][:-1], data["ee_pose"][:-1]], axis=1))
                self.x_actions.extend(data["ee_pose"][1:])

            elif self.action_space == "endeff-window":
                # Use a sliding window of size 10 to predict poses...
                idxs = sliding_window_view(np.arange(len(data["rgb"])), window_shape=10)[:, (0, -1)]
                self.rgbs.extend(data["rgb"][idxs[:, 0]])
                self.proprios.extend(np.concatenate([data["q"][idxs[:, 0]], data["ee_pose"][idxs[:, 0]]], axis=1))
                self.x_actions.extend(data["ee_pose"][idxs[:, 1]])

            elif self.action_space in ["joint-delta", "joint-delta-norm"]:
                # =>> Note -- "delta" fields are offset by 1 timestep to the right!
                self.rgbs.extend(data["rgb"][:-1])
                self.proprios.extend(np.concatenate([data["q"][:-1], data["ee_pose"][:-1]], axis=1))
                self.x_actions.extend(data["delta_q"][1:])

            elif self.action_space == "joint":
                # =>> Note -- "delta" fields are offset by 1 timestep to the right!
                self.rgbs.extend(data["rgb"][:-1])
                self.proprios.extend(np.concatenate([data["q"][:-1], data["ee_pose"][:-1]], axis=1))
                self.x_actions.extend(data["q"][1:])

            elif "grip" in self.action_space:
                raise NotImplementedError(f"We don't handle gripper anymore - `{self.action_space}` is invalid!")

            else:
                raise NotImplementedError(f"Support for action space `{self.action_space}` not yet implemented!")

        # Tensorize
        self.rgbs = torch.stack([self.transform(img) for img in self.rgbs])
        self.proprios = torch.Tensor(np.stack(self.proprios))
        self.x_actions = torch.Tensor(np.stack(self.x_actions))

        # Update dimensions
        self.proprioceptive_dim = self.proprios.shape[-1]
        self.x_action_dim = self.x_actions.shape[-1]

        # Compute full set of actions
        self.actions = self.x_actions
        self.action_dim = self.actions.shape[-1]

        # Normalize each "action"
        if "-norm" in self.action_space:
            self.action_mean, self.action_std = self.actions.mean(dim=0), self.actions.std(dim=0)
            self.actions = (self.actions - self.action_mean) / self.action_std

        # Assertion
        assert len(self.rgbs) == len(self.proprios) == len(self.actions), "We don't have an equal # of samples!"

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rgbs[index], self.proprios[index], self.actions[index]

    def __len__(self) -> int:
        return len(self.proprios)


def get_dataset(state_space: str, action_space: str, data: Path, n_train: int, n_val: int) -> Tuple[Dataset, Dataset]:
    """Create PyTorch Dataset Objects depending on the dataset modality & data path."""
    train_data, val_data = [], []
    for task in data.iterdir():
        if task.is_file():
            continue

        # Parse out RGB files...
        rgb_path = task / "playback-rgb"
        demos = sorted(list(rgb_path.iterdir()), key=lambda x: int(str(x).split("-")[-1].split(".")[0]))

        # Get valid indices...
        train_demos, val_demos = demos[:-n_val], demos[-n_val:]
        assert len(train_demos) == n_train and len(val_demos) == n_val
        for d in val_demos:
            assert "06-13" not in str(d)

        # Create splits & add to the respective datasets.
        train_data.extend(train_demos)
        val_data.extend(val_demos)

    if state_space == "rgb":
        return RGBDataset(train_data, action_space), RGBDataset(val_data, action_space)
    else:
        raise NotImplementedError(f"State Space `{state_space}` not yet implemented!")
