import torch
from torch.utils.data import DataLoader

from algos import BaseAlgorithm


class MaxEntIRL:
    def __init__(
        self,
        model,
        model_kwargs,
        save_dir,
        device,
        expert_policy=None,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        policy_cls="LinearModel",
    ) -> None:
        super().__init__(
            model,
            model_kwargs,
            save_dir,
            device,
            expert_policy=expert_policy,
            lr=lr,
            optimizer=optimizer,
            policy_cls=policy_cls,
        )
        self.env = env
        self.device = device
        self.feature_size = feature_size
        # change to MLP?
        self.features =  nn.Linear(env.obs_dim, feature_size)
        self.optim = torch.optim.adam([self.r.parameters(), self.features.parameters()], lr=args.lr)
        self.loss = nn.C

    def get_feature_expectations(self, demonstrations):
        """
        Get the expected feature expectation for trajectories.
        """
        feature_expectations = torch.zeros((self.feature_size,)).to(self.device)
        for demo in demonstrations:
            for (obs, act) in zip(demo["obs"], demo["act"]):
                self.optim.zero_grad()
                feature_expectations += self.features(obs)
        feature_expectations /= len(demonstrations)
        return feature_expectations
                
    def get_reward(self, reward, features):
        """
        Get the reward for a given feature vector.
        """
        r = (reward*features).sum(-1)
        return r
    
    
    def run(self, train_data, val_data, args, env=None, robosuite_cfg=None) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        if args.robosuite:
            robosuite_cfg["input_device"].start_control()

        # Train => Handle Ensemble...
        if self.is_ensemble:
            for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                self.train(model, optimizer, train_loader, val_loader, args)
        else:
            self.train(self.model, self.optimizer, train_loader, val_loader, args)

        # Save Metrics
        self._save_metrics()
