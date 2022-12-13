import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size = obs_dim, act_dim, hidden_size

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs):
        mean = self.layers(obs)
        std = torch.ones_like(mean) * self.log_std.exp()
        return Normal(mean, std)

    def get_action(self, obs, deterministic=True):
        dist = self.forward(obs)
        if deterministic:
            return dist.mean
        else:
            return dist.sample()
    
    def get_compatibility(self, obs, act, type="likelihood"):
        if type == "likelihood":
            dist = self.forward(obs)
            likelihood = torch.clamp(dist.log_prob(act), min=-1e4, max=1e4)
            return likelihood.sum()
        elif type == "entropy":
            dist = self.forward(obs)
            entropy = dist.entropy()
            return entropy
        else:
            raise NotImplementedError(f"Compatibility {type} not implemented")
