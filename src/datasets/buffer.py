from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import MAX_BUFFER_SIZE


class BufferDataset(Dataset):
    def __init__(self, data, max_size=MAX_BUFFER_SIZE, shuffle=True) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.w = None
        self.obs, self.act, self.curr_size = self.split_data(data, shuffle=shuffle)
        self.w = torch.ones(self.max_size) 
        self.ptr += self.curr_size

    def split_data(self, data, shuffle) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = data["obs"]
        act = data["act"]
        curr_size = len(obs)

        if shuffle:
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            obs = obs[idxs]
            act = act[idxs]

        obs = torch.cat([obs, torch.zeros(max(self.max_size - len(obs), self.max_size), obs.shape[1])])
        act = torch.cat([act, torch.zeros(max(self.max_size - len(act), self.max_size), act.shape[1])])
        return obs, act, curr_size

    def clear_buffer(self):
        self.obs = torch.zeros(self.max_size, self.obs.shape[1])
        self.act = torch.zeros(self.max_size, self.act.shape[1])
        self.w = torch.ones(self.max_size)
        self.curr_size = 0
        self.ptr = 0

    def update_buffer(self, obs, act, w=None) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        if w is not None:
            self.w[self.ptr] = w
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def get_obs_stats(self):
        mean = self.obs.mean(dim=0)
        std = self.obs.std(dim=0)
        return mean, std
    
    def __len__(self) -> int:
        return self.curr_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (state (i.e., observation), action) tuple
        return self.obs[idx], self.act[idx], self.w[idx]


class BufferSequenceDataset(BufferDataset):
    def split_data(self, data, shuffle) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = data["obs"]
        act = data["act"]
        curr_size = len(obs)
        if shuffle:
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            obs = obs[idxs]
            act = act[idxs]
        obs = torch.cat([obs, torch.zeros(max(self.max_size - len(obs), self.max_size), obs.shape[1], obs.shape[2])])
        act = torch.cat([act, torch.zeros(max(self.max_size - len(act), self.max_size), act.shape[1], act.shape[2])])
        return obs, act, curr_size

    def clear_buffer(self):
        self.obs = torch.zeros(self.max_size, self.obs.shape[1])
        self.act = torch.zeros(self.max_size, self.act.shape[1])
        self.w = torch.ones(self.max_size)
        self.curr_size = 0
        self.ptr = 0

    def update_buffer(self, obs, act, w = None) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        if w is not None:
            self.w[self.ptr] = w
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def get_obs_stats(self):
        raise NotImplementedError
    
    def __len__(self) -> int:
        return self.curr_size
