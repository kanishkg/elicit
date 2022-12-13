import time
import pickle
import os

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



class ThriftyDagger(BaseAlgorithm):
    def __init__(
        self,
        model,
        model_kwargs,
        save_dir,
        device,
        expert_policy,
        q_model=None,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        q_optimizer=torch.optim.Adam,
        q_lr=1e-3,
        beta=0.9,
        max_num_labels=1000,
        policy_cls="LinearModel",
    ) -> None:

        super().__init__(model, model_kwargs, save_dir, device, expert_policy=expert_policy, lr=lr, optimizer=optimizer, policy_cls=policy_cls)

        self.q_model = q_model
        self.q_optimizer = q_optimizer
        self.q_optimizers = [
                self.q_optimizer(self.q_model.models[i].parameters(), lr=q_lr) for i in range(len(self.q_model.models))
            ]
        self.max_num_labels = max_num_labels
        self.num_labels = 0
        self.expert_mode = False

    def _save_data(self, data, epoch):
        save_path = os.path.join(self.save_dir, f"data_epoch{epoch}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def train_risk(self, args, epoch, num_episodes):
        if args.best_epoch:
            # eval model with the best validation loss
            model_path = os.path.join(args.save_dir, "model_best.pt")
            ckpt = torch.load(model_path, map_location=self.device)
            if args.num_models > 1:
                for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                    ensemble_model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        if num_test_episodes > 0:
            test_agent(t) # collect samples offline from pi_R
            data = pickle.load(open('test-rollouts.pkl', 'rb'))
            qbuffer.fill_buffer(data)
            os.remove('test-rollouts.pkl')
            q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
            q_optimizer = Adam(q_params, lr=pi_lr)
            loss_q = []
            for _ in range(bc_epochs):
                for i in range(grad_steps * 5):
                    batch = qbuffer.sample_batch(batch_size // 2, pos_fraction=0.1)
                    loss_q.append(update_q(batch, timer=i))
        pass

    def switch_mode(self, act, obs, robosuite_cfg=None, env=None, action_lim=None):
        if not self.expert_mode:
            # switch to expert if obs is sufficiently novel or there is a low probability of success
            if self.model.get_uncertainty(obs) > NOVELTY_THRESH:
                return True
            if self.q.safety(obs, act) < RISK_THRESH:
                return True
        else:
            # switch to robot if the predicted actions are close to the expert and the the probability of success is high
            pred_act = self.model.get_action(obs)
            pred_act = torch.clamp(a, min=-action_lim, max=action_lim)
            a = torch.clamp(a, min=-action_lim, max=action_lim)
            if sum((pred_act - act) ** 2) < SWITCH2ROBOT_THRESH and self.q.safety(obs, act) > SWITCH2ROBOT_RISK_THRESH:
                return True
        return False


    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        raise NotImplementedError
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)

        mean, std = train_data.get_obs_stats()
        mean = mean.to(self.device)
        std = std.to(self.device)
        action_lim = env.action_space.high[0]

        # get interactive reaching data 
        self.train_demos = get_policy_data(args.data_path) 


        for epoch in range(args.dagger_epochs+1):
            robosuite_cfg["input_device"].start_control()
            # Train policy
            self._reset_model()
            self._setup_optimizer()
            if self.is_ensemble:
                for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                    self.train(model, optimizer, train_loader, val_loader, args, model_idx=i, mean=mean, std=std)
            else:
                self.train(self.model, self.optimizer, train_loader, val_loader, args, mean=mean, std=std)
            
            self.train_risk()

            if args.best_epoch:
                model_path = os.path.join(self.save_dir, "model_best.pt")
                ckpt = torch.load(model_path, map_location=self.device)
                if args.num_models > 1:
                    for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                        ensemble_model.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(ckpt["model"])
            self.model.eval()

            # Roll out trajectories
            if epoch < args.dagger_epochs and self.num_labels < self.max_num_labels:
                new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout, args)
            if self.num_labels >= self.max_num_labels:
                break

            # log dagger epochs against num switches on wandb
            if len(new_data) > 0:
                self._save_data(new_data, epoch)
                for d, demo in enumerate(new_data):
                    # self.train_demos.append(demo)
                    for p, (obs, act) in enumerate(zip(demo["obs"], demo["act"])):
                        train_data.update_buffer(obs, act)
                        self.num_labels += 1
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

            mean_uncertainty = torch.mean(torch.stack(self.interact_uncertainty))
            std_uncertainty = torch.std(torch.stack(self.interact_uncertainty))
            wandb.log({"dagger_epoch": epoch, "num_switches": self.num_switches, 'interact_uncertainty_mean': mean_uncertainty, 'interact_uncertainty_std': std_uncertainty})

