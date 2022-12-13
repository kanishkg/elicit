import os
import pickle
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import wandb

from algos import BaseAlgorithm
from util.int_utils import get_demo_compatibility, count_bad_samples, count_good_samples


class Dagger(BaseAlgorithm):
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
        use_indicator_beta=False,
        max_num_labels=1000,
        policy_cls="LinearModel",
    ) -> None:

        super().__init__(model, model_kwargs, save_dir, device, expert_policy=expert_policy, lr=lr, optimizer=optimizer, policy_cls=policy_cls)

        self.use_indicator_beta = use_indicator_beta
        self.beta = self._init_beta(beta)
        self.max_num_labels = max_num_labels
        self.num_labels = 0

    def _init_beta(self, beta):
        if self.use_indicator_beta and (beta != 1.0):
            raise ValueError(f"If use_indicator_beta is True, beta must be 1.0, but got beta={beta}!")
        return beta

    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, auto_only=False):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs = env.reset()
            done = False
            reached_success = False
            obs, act = [], []
            while not done and self.num_labels < self.max_num_labels:
                if not auto_only:
                    a_target = self.expert_policy.act(curr_obs)
                    a_target = a_target.to(self.device)
                    a = self.beta * a_target + (1 - self.beta) * self.model.get_action(curr_obs).detach()
                    self.num_labels += 1
                    act.append(a_target.cpu())
                else:
                    # TODO: add state and xml saving like base algo
                    a = self.model.get_action(curr_obs).detach()
                    act.append(a.cpu())
                next_obs, success, done, _ = env.step(a)
                obs.append(curr_obs.cpu())

                # Document whether or not success was reached, but continue
                # rolling out until done (DAgger rolls out until max trajectory length
                # is reached, not until success is reached)
                if success:
                    reached_success = True
                curr_obs = next_obs

            demo = {"obs": obs, "act": act, "success": reached_success}
            data.append(demo)
            env.close()

            if self.num_labels >= self.max_num_labels:
                break

        return data

    def _save_data(self, data, epoch):
        save_path = os.path.join(self.save_dir, f"data_epoch{epoch}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        # initial policy training + dagger interactive training
        new_data = []
        if robosuite_cfg is None: 
            low, high = env.action_space
            action_lim = high
        else:
            action_lim = env.action_space.high[0]
        

        dagger_epochs = args.dagger_epochs if args.init_dagger_model else args.dagger_epochs + 1
        if args.offline_policy:
            if args.compatibility_type == "random":
                random.shuffle(self.expert_policy)
            elif args.compatibility_type == "bad":
                assert args.init_dagger_model
                ckpt = torch.load(args.model_path, map_location=self.device)
                if args.num_models > 1:
                    for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                        ensemble_model.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(ckpt["model"])
                self.model.eval()
                num_bad = []
                num_good = []
                for i, demo in enumerate(self.expert_policy):
                    comp_matrix = get_demo_compatibility(self.model, demo, self.device, action_lim=action_lim)
                    bad = count_bad_samples(comp_matrix, args)
                    # negative because we want descending order 
                    good = -count_good_samples(comp_matrix, args)
                    num_bad.append(bad)
                    num_good.append(good)
                # Sort based expert demos based on number of bad samples, break ties by number of good samples
                self.expert_policy = [x for _, _, x in sorted(zip(num_bad, num_good, self.expert_policy), key=lambda pair: (pair[0], pair[1]))]
                num_good = [x for _, x in sorted(zip(num_bad, num_good))]
                self.expert_policy = self.expert_policy[: dagger_epochs*args.trajectories_per_rollout*2]
                num_good = num_good[: dagger_epochs*args.trajectories_per_rollout*2]
                self.expert_policy = [x for _, x in sorted(zip(num_good, self.expert_policy), key=lambda pair: pair[0])]
            elif args.compatibility_type == "good":
                assert args.init_dagger_model
                ckpt = torch.load(args.model_path, map_location=self.device)
                if args.num_models > 1:
                    for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                        ensemble_model.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(ckpt["model"])
                self.model.eval()
                num_bad = []
                num_good = []
                for i, demo in enumerate(self.expert_policy):
                    comp_matrix = get_demo_compatibility(self.model, demo, self.device, action_lim=action_lim)
                    bad = count_bad_samples(comp_matrix, args)
                    # negative because we want descending order 
                    good = -count_good_samples(comp_matrix, args)
                    num_bad.append(bad)
                    num_good.append(good)
                self.expert_policy = [x for _, x in sorted(zip(num_good, self.expert_policy), key=lambda pair: pair[0])]
            else:
                random.shuffle(self.expert_policy)

        start_id = 0
        for epoch in range(dagger_epochs):
            if args.robosuite:
                robosuite_cfg["input_device"].start_control()

            # Roll out trajectories
            if epoch > 0 or args.init_dagger_model:
                if args.offline_policy:
                    # simple aggregation where demos are  being added to the buffer
                    # similar to behavior cloning but now we check compataibility with existing policy
                    # start_id = max(0, epoch-1)
                    if args.compatibility_type == "random" or args.compatibility_type == "box":
                        new_traj = self.expert_policy[start_id*args.trajectories_per_rollout: (start_id+1)*args.trajectories_per_rollout]
                        new_samples = 0
                        new_data = []
                        for d, demo in enumerate(new_traj):
                            if new_samples+len(demo["obs"]) < args.samples_per_epoch:
                                new_data.append(demo)
                                new_samples += len(demo["obs"])
                            else:
                                demo["obs"] = demo["obs"][:args.samples_per_epoch-new_samples]
                                demo["act"] = demo["act"][:args.samples_per_epoch-new_samples]
                                new_data.append(demo)
                                new_samples += len(demo["obs"])
                                start_id += d
                                break
                    else:
                        new_samples = 0
                        new_data = []
                        for d, demo in enumerate(self.expert_policy[start_id:]):
                            if new_samples+len(demo["obs"]) < args.samples_per_epoch:
                                new_data.append(demo)
                                new_samples += len(demo["obs"])
                            else:
                                demo["obs"] = demo["obs"][:args.samples_per_epoch-new_samples]
                                demo["act"] = demo["act"][:args.samples_per_epoch-new_samples]
                                new_data.append(demo)
                                new_samples += len(demo["obs"])
                                start_id += d
                                break
                    wandb.run.summary[f"new_samples"] = new_samples

                else:
                    raise NotImplementedError("online collection not implemented")
            # Add new data to training dataset only after first iteration, as we want to train on the initial dataset
            print(f"Adding {len(new_data)} new trajectories to training dataset")
            if len(new_data) > 0:
                if args.check_compatibility:
                    if args.init_dagger_model and epoch == 0:
                        ckpt = torch.load(args.model_path, map_location=self.device)
                    else:
                        model_path = os.path.join(self.save_dir, "model_best.pt")
                        ckpt = torch.load(model_path, map_location=self.device)
                    if args.num_models > 1:
                        for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                            ensemble_model.load_state_dict(state_dict)
                    else:
                        self.model.load_state_dict(ckpt["model"])
                    self.model.eval()
                    num_skipped, num_seen = 0, 0
                    for d, demo in enumerate(new_data):
                        num_seen += len(demo["obs"])
                        if not args.compatibility_type == "random":
                            compatibility = get_demo_compatibility(self.model, demo, self.device, action_lim=action_lim)
                        for p, (obs, act) in enumerate(zip(demo["obs"], demo["act"])):
                            w = None 
                            if not args.compatibility_type == "random":
                                l = compatibility[p, 0]
                                e = compatibility[p, 1]
                            if args.compatibility_type == "weighted":
                                if e < 0.1:
                                    w = max(0.6 - l, 0)
                                else:
                                    w = e/0.1 * (0.8-l)/0.4
                            # c = (e**2) + (l-0.2)**2
                            # t = args.entropy_filter/args.likelihood_filter**2
                            elif args.compatibility_type == "line":
                                if e < 0.5*l - 0.1:
                                    num_skipped += 1
                                    continue
                            elif args.compatibility_type == "box" or args.compatibility_type == "good":
                                # remove demos with familiar states and unlikely actions 
                                if (l > args.likelihood_filter) and (e < args.entropy_filter):
                                    num_skipped += 1
                                    continue
                            elif args.compatibility_type == "topbox":
                                if e > args.entropy_filter:
                                    num_skipped += 1
                                    continue
                            elif args.compatibility_type == "leftbox":
                                if e > args.entropy_filter and l > args.likelihood_filter:
                                    num_skipped += 1
                                    continue
                            elif args.compatibility_type == "rightbox":
                                if e > args.entropy_filter and l < args.likelihood_filter:
                                    num_skipped += 1
                                    continue
                            train_data.update_buffer(obs, act, w)
                    print(f"Skipped {num_skipped} demos out of {num_seen}")
                    wandb.run.summary[f"num_skipped"] = num_skipped 

                self._save_data(new_data, epoch)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size)

            # If max number of expert labels exceeded, break
            if self.num_labels > self.max_num_labels:
                break

            # Reset the model and optimizer for retraining
            if epoch > 0 or args.init_dagger_model:
                self._reset_model()
                self._setup_optimizer()

            # Retrain policy
            if self.is_ensemble:
                for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                    self.train(model, optimizer, train_loader, val_loader, args, model_idx=i)
            else:
                self.train(self.model, self.optimizer, train_loader, val_loader, args)

            if self.use_indicator_beta:
                # Set beta to 0 after first interaction
                if epoch == 1:
                    self.beta = 0
            else:
                # Beta decays exponentially
                self.beta *= self.beta

        # TODO: add more dagger-specific metrics (num switches between robot/human)
        # self._save_metrics()
