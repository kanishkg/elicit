import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, num_layers=2,
                  bidirectional=False, seq_len=10, layer_norm=False, dropout=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(obs_dim, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, batch_first=True, dropout=dropout)
        if not bidirectional:
            self.out_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(),
                nn.Linear(hidden_size, act_dim), nn.Tanh())
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(hidden_size*2, act_dim), nn.Tanh())

    def forward(self, obs):
        out, _ = self.rnn(obs)
        acts = self.out_mlp(out)
        return acts

    def get_action(self, obs, state):
        # expects obs to be of shape B x T x D or B x D
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
        # state is a tuple of (hidden, cell)
        # hidden, cell is of size B x L x H -> L x B x H
        # take transpose to get L x B x H
        hs = state[0].permute(1, 0, 2)
        cs = state[1].permute(1, 0, 2)
        out, new_state = self.rnn(obs, (hs, cs)) 
        acts = self.out_mlp(out)
        # acts is of shape B x T x A
        # new state is a tuple of (hidden, cell)
        # with hidden and cell of size D*L x B x H
        hs = new_state[0].permute(1, 0, 2)
        cs = new_state[1].permute(1, 0, 2)
        return acts, (hs, cs)

    def get_compatibility(self, obs, act, state, type="likelihood", action_lim=1.0, reduction="all"):
        raise NotImplementedError("Compatibility not implemented")

    def get_uncertainty(self, obs, state):
        raise NotImplementedError("Uncertainty not implemented")