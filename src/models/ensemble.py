import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, model_kwargs, device, num_models=5, model_type=None) -> None:
        super().__init__()
        self.num_models = num_models
        self.device = device
        self.models = [model_type(**model_kwargs).to(device) for _ in range(num_models)]

    def forward(self, obs):
        # obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # No grad since we don't want to backpropagate over taking the average of the ensemble
        with torch.no_grad():
            acts = []
            for model in self.models:
                acts.append(model(obs).detach())
            return torch.mean(torch.stack(acts), dim=0)

    def get_action(self, obs, state=None):
        if state is not None:
            acts, new_h, new_c = [], [], []
            for i, model in enumerate(self.models):
                hs, cs = state[0][i], state[1][i]
                a, new_state = model.get_action(obs, (hs, cs))
                acts.append(a.detach())
                new_h.append(new_state[0].detach())
                new_c.append(new_state[1].detach())
            new_h = torch.stack(new_h)
            new_c = torch.stack(new_c)
            return torch.mean(torch.stack(acts), dim=0), (new_h, new_c)
        else:
            return self.forward(obs)

    def get_compatibility(self, obs, act, type="likelihood", action_lim=1.0, reduction="all"):
        if type == "likelihood":
            with torch.no_grad():
                compatibility = []
                for model in self.models:
                    act = torch.clamp(act, min=-action_lim, max=action_lim)
                    pred_act = model(obs).detach()
                    pred_act = torch.clamp(pred_act, min=-action_lim, max=action_lim)
                    compatibility.append(((pred_act-act)**2).mean(dim=-1))
                if reduction == "all":
                    # average over all models and trajectories
                    compatibility_score = torch.mean(torch.stack(compatibility))
                elif reduction == "obs":
                    # average across ensemble models. Return a list of compatibility scores for each observation
                    compatibility_score = torch.mean(torch.stack(compatibility, dim=0), dim=0)
                return compatibility_score
        elif type == "entropy":
            with torch.no_grad():
                compatibility = []
                for model in self.models:
                    compatibility.append(model(obs).detach())
                if reduction == "all":
                    # average over all models and trajectories
                    compatibility_score = torch.mean(torch.std(torch.stack(compatibility), dim=0))
                elif reduction == "obs":
                    # get std across ensemble models -> mean across dims. Return a list of compatibility scores for each observation
                    compatibility_score = torch.std(torch.stack(compatibility, dim=0), dim=0).mean(dim=-1)
                return compatibility_score
        else:
            raise NotImplementedError(f"Compatibility {type} not implemented")
    
    def get_uncertainty(self, obs, action_lim=1.0):
        with torch.no_grad():
            uncertainty = [] 
            obs = obs.unsqueeze(0)
            for model in self.models:
                uncertainty.append(model(obs).detach())
            uncertainty = torch.mean(torch.std(torch.stack(uncertainty, dim=0), dim=0))
        return uncertainty