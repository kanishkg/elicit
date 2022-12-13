import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, obs_dim, act_dim, scale=1.0, normalize=False, use_bias=False):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim, bias=use_bias)
        self.scale = scale
        self.normalize = normalize

    def forward(self, obs):
        output = self.linear(obs)
        with torch.no_grad():
            if self.normalize:
                if len(output.shape) > 1:
                    output /= torch.norm(output, dim=1).view(-1, 1)
                else:
                    output /= torch.norm(output)
            output *= self.scale
        return output

    def get_action(self, obs):
        return self.forward(obs)
