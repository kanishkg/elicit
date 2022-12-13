import torch
from torch.utils.data import DataLoader

from algos import BaseAlgorithm


class BC(BaseAlgorithm):
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

    def run(self, train_data, val_data, args, env=None, robosuite_cfg=None) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        if args.robosuite:
            robosuite_cfg["input_device"].start_control()

        # Train => Handle Ensemble...
        if self.is_ensemble:
            for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                self.train(model, optimizer, train_loader, val_loader, args, model_idx=i)
        else:
            self.train(self.model, self.optimizer, train_loader, val_loader, args)
        # Save Metrics
        # self._save_metrics()
