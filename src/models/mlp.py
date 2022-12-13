from mimetypes import init
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, layer_norm=False, dropout=0.0):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size = obs_dim, act_dim, hidden_size

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialize weights with xavier uniform 
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs):
        return self.layers(obs)

    def get_action(self, obs):
        return self.forward(obs)

    def get_compatibility(self, obs, act, type="likelihood", action_lim=1.0):
        if type == "likelihood":
            pred_act = self.forward(obs).detach()
            pred_act = torch.clamp(pred_act, min=-action_lim, max=action_lim)
            compatibility_score = ((pred_act-act)**2).mean()
        else:
            raise NotImplementedError(f"Compatibility {type} not implemented") 
        return compatibility_score