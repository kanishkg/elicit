from mimetypes import init
import torch
import torch.nn as nn


class qMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size = obs_dim, act_dim, hidden_size

        self.layers = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialize weights with xavier uniform 
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.layers(x)
        return q.squeeze(-1)

class qEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, num_models, device):
        super().__init__()
        self.models = [qMLP(obs_dim, act_dim, hidden_size).to(device) for _ in range(num_models)]
        self.device = device

    def forward(self, obs, act):
        q = torch.stack([model(obs, act) for model in self.models], dim=0)
        return q.mean(dim=0)

    def safety(self, obs, act, action_lim=1.0):

        with torch.no_grad():
            q = torch.stack([model(obs, act) for model in self.models], dim=0)
            safety = torch.min(q)
        return safety

