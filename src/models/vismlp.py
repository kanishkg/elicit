from mimetypes import init
import torch
import torch.nn as nn
from torchvision.models import resnet18


class VisMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, layer_norm=False, dropout=0.0, pretrained=True):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size = obs_dim, act_dim, hidden_size
        resnet = resnet18(pretrained=True)
        # till adaptive avg pooling layer, out shape is 512
        self.cnn_encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU())

        self.act = nn.Sequential(
            nn.Linear(hidden_size+512, hidden_size),
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
        img_obs = obs[0]
        state_obs = obs[1]
        img_embed = self.cnn_encoder(img_obs)
        state_embed = self.obs_encoder(state_obs)
        embed = torch.cat([img_embed, state_embed], dim=-1)
        out = self.act(embed)
        return out

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