import time
import os
import pickle
import tqdm

import numpy as np
import torch
import wandb
try:
    from robosuite.utils.input_utils import input2action
except ImportError:
    print("Could not import robosuite.utils.input_utils.input2action")
    
from torch.utils.data import DataLoader

from algos import BaseAlgorithm
from util.int_utils import get_env, visualize_demo, load_model, get_demo_compatibility, count_bad_samples, sample_trajectory, show_bad_samples, get_nearest
from datasets.util import get_policy_data


class HGDagger(BaseAlgorithm):
    def __init__(
        self,
        model,
        model_kwargs,
        save_dir,
        device,
        expert_policy,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        beta=0.9,
        max_num_labels=10000,
        policy_cls="LinearModel",
    ) -> None:

        super().__init__(model, model_kwargs, save_dir, device, expert_policy=expert_policy, lr=lr, optimizer=optimizer, policy_cls=policy_cls)


    def _save_data(self, data, epoch):
        save_path = os.path.join(self.save_dir, f"data_epoch{epoch}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _switch_mode(self, act, obs=None, robosuite_cfg=None, env=None, action_lim=None):
        # If in robot mode, need to check for input from human
        # A 'Z' keypress (action elem 3) indicates a mode switch
        if act is None:
            for _ in range(10):
                act, _ = input2action(
                    device=robosuite_cfg["input_device"],
                    robot=robosuite_cfg["active_robot"],
                    active_arm=robosuite_cfg["arm"],
                    env_configuration=robosuite_cfg["env_config"],
                )
                env.render()
                if act is None:
                    action = "reset"
                    robosuite_cfg['input_device']._reset_state = 0
                    # self._enabled = True
                    return action
                time.sleep(0.001)
                if act[3] != 0:  # 'Z' is pressed
                    break
        return act[3] != 0

    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        mean, std = train_data.get_obs_stats()
        mean = mean.to(self.device)
        std = std.to(self.device)
        action_lim = env.action_space.high[0]

        # get interactive reaching data 
        with open(args.data_path, "rb") as f:
            self.train_demos = pickle.load(f)


        new_data = []
        dagger_epochs = args.dagger_epochs if args.init_dagger_model else args.dagger_epochs + 1
        for epoch in range(dagger_epochs):
            robosuite_cfg["input_device"].start_control()


            if args.init_dagger_model and epoch == 0:
                ckpt = torch.load(args.model_path, map_location=self.device)
            else:
                # Train policy
                self._reset_model()
                self._setup_optimizer()
            if args.num_models > 1:
                for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                        ensemble_model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            if not args.init_dagger_model or epoch > 0:
                self.model.train()
                if self.is_ensemble:
                    for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                        self.train(model, optimizer, train_loader, val_loader, args, model_idx=i)
                else:
                    self.train(self.model, self.optimizer, train_loader, val_loader, args)

                if args.best_epoch:
                    model_path = os.path.join(self.save_dir, "model_best.pt")
                    ckpt = torch.load(model_path, map_location=self.device)
                    if args.num_models > 1:
                        for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                            ensemble_model.load_state_dict(state_dict)
                    else:
                        self.model.load_state_dict(ckpt["model"])
                self.model.eval()

            if self.num_labels >= self.max_num_labels:
                break
            # Roll out trajectories
            if epoch < args.dagger_epochs and self.num_labels < self.max_num_labels:
                new_data, all_new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout, args)

            # log dagger epochs against num switches on wandb
            if len(new_data) > 0:
                self._save_data(new_data, f"{epoch}_human")
                self._save_data(all_new_data, f"{epoch}_all")
                for d, demo in enumerate(new_data):
                    # self.train_demos.append(demo)
                    for p, (obs, act) in enumerate(zip(demo["obs"], demo["act"])):
                        train_data.update_buffer(obs, act)
                        # self.num_labels += 1
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

            mean_uncertainty = torch.mean(torch.stack(self.interact_uncertainty))
            std_uncertainty = torch.std(torch.stack(self.interact_uncertainty))
            wandb.log({"dagger_epoch": epoch, "num_switches": self.num_switches, 'interact_uncertainty_mean': mean_uncertainty, 'interact_uncertainty_std': std_uncertainty})
