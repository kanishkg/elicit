from random import gauss
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent


class MDN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, n_components=2):
        """Mixture Density Networks -- fits a K-Mixture of Gaussians, where k = n_components..."""
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size, self.n_components = obs_dim, act_dim, hidden_size, n_components

        # Gaussian --> Predict means, diagonalized variances for *each* component
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.gaussian_mean = nn.Linear(hidden_size, act_dim * n_components)
        self.gaussian_scale = nn.Linear(hidden_size, act_dim * n_components)
        self.pi_logits = nn.Linear(hidden_size, n_components)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialize weights with xavier uniform 
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, obs):
        hidden = self.net(obs)
        mean, std = self.gaussian_mean(hidden), self.gaussian_scale(hidden)
        logits = self.pi_logits(hidden)
        # similar to robomimic
        mean = torch.clamp(mean, min=-9.0, max=9.0)
        std = torch.clamp(std, min=0.007, max=7.5)
        mean, std = torch.tanh(mean), torch.exp(std)
        mean, std = mean.view(-1, self.n_components, self.act_dim), std.view(-1, self.n_components, self.act_dim)
        categorical, gaussians =  Categorical(logits=logits), Normal(loc=mean, scale=std)
        gaussians = Independent(gaussians, 1)
        gmm = MixtureSameFamily(categorical, gaussians)
        return gmm, categorical, gaussians

    def get_action(self, obs, deterministic=False):
        # Sampling Code
        # Use non-deterministic at evaluation time? with low std?
        hidden = self.net(obs)
        mean, std = self.gaussian_mean(hidden), self.gaussian_scale(hidden)
        mean, std = mean.view(-1, self.n_components, self.act_dim), std.view(-1, self.n_components, self.act_dim)
        logits = self.pi_logits(hidden)
        # similar to robomimic
        mean = torch.clamp(mean, min=-9.0, max=9.0)
        std = torch.clamp(std, min=0.007, max=7.5)
        mean, std = torch.tanh(mean), torch.exp(std)

        if deterministic:
            categorical, gaussians =  Categorical(logits=logits), Normal(mean, std)
            gaussians = Independent(gaussians, 1)
            gmm = MixtureSameFamily(categorical, gaussians)
            mean = gmm.mean
        else:
            # low noise evaluation
            low_std = torch.ones_like(std) * 1e-4
            categorical, gaussians =  Categorical(logits=logits), Normal(mean, low_std)
            gaussians = Independent(gaussians, 1)
            gmm = MixtureSameFamily(categorical, gaussians)
            mean = gmm.sample()
        return mean

    def get_compatibility(self, obs, act, type="likelihood", action_lim=None, reduction="all"):
        #TODO change for gmm
        if type == "likelihood":
            gmm, pi_dist, gaussian_dist = self.forward(obs)
            log_prob = gmm.log_prob(act)
            if reduction == "all":
                return log_prob.mean()
            elif reduction == "obs":
                return log_prob 

        elif type == "entropy":
            gmm, pi_dist, gaussian_dist = self.forward(obs)
            std = torch.sqrt(gmm.variance)
            if reduction == "all":
                return std.mean() 
            elif reduction == "obs":
                return std.mean(dim=-1)
            # TODO - check if this is correct
            # simple upper bound on entropy (8) in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4648062
            # can be improved by using (11) instead of (8) 
            # pi_ent = pi_dist.entropy()
            # weighted_gauss_entropy = pi_dist.probs * gaussian_dist.entropy()
            # entropy_u = pi_ent + weighted_gauss_entropy.sum(dim=1)
            # if reduction == "all":
            #     return entropy_u.mean() 
            # elif reduction == "obs":
            #     return entropy_u
        else:
            raise NotImplementedError(f"Compatibility {type} not implemented")