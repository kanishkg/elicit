import os
import pickle
from collections import defaultdict
import time
import random

import pandas as pd
import numpy as np
import torch

from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt

from tqdm import tqdm
import wandb

from models import Ensemble
from util import init_model, setup_robosuite
from util.int_utils import parse_obs, visualize_demo, load_model, get_demo_compatibility, count_bad_samples, sample_trajectory, show_bad_samples, get_nearest


class BaseAlgorithm:
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
        max_num_labels=10000,
        max_env_steps=10000, 
    ) -> None:

        self.model = model
        self.model_kwargs = model_kwargs
        self.save_dir = save_dir
        self.device = device
        self.expert_policy = expert_policy
        self.lr = lr

        self.model_type = type(model)
        self.optimizer_type = optimizer
        self.is_ensemble = self.model_type == Ensemble
        self.policy_cls = policy_cls

        # Setup Optimizer, Metrics, and set Loss Function
        self._setup_optimizer()
        self._setup_metrics()
        self.loss_fn = self._get_loss_fn()
        self.best_ckpt_dict = dict()
        self.num_switches = 0
        self.interact_uncertainty = []
        self.expert_mode = False
        self.train_demos = []
        self.max_num_labels = max_num_labels
        self.num_labels = 0
        self.env_steps = 0
        self.max_env_steps = max_env_steps
        if self.is_ensemble:
            self.best_ckpt_dict = {"models": [],
                                "optimizers": [],
                                "epoch": []}
            
    def _get_loss_fn(self):
        if self.policy_cls in ["LinearModel", "MLP"]:

            def loss_fn(model_nn, observation, action, weights):
                predicted_action = model_nn(observation)
                return torch.mean(weights*torch.sum((action - predicted_action) ** 2, dim=1))

            return loss_fn

        elif self.policy_cls in ["GaussianMLP"]:

            def loss_fn(model_nn, observation, action, weights):
                predicted_dist = model_nn(observation)
                log_prob = predicted_dist.log_prob(action).sum(dim=1)
                # Minimize negative log likelihood...
                return -torch.mean(log_prob)
            return loss_fn

        elif self.policy_cls in ["MDN"]:
            def loss_fn(model_nn, observation, action, weights):
                gmm, _, _ = model_nn(observation)
                log_prob = gmm.log_prob(action).mean()
                return -log_prob
            return loss_fn
        elif self.policy_cls in ["RNN"]:
            def loss_fn(model_nn, observation, action, weights):
                predicted_action = model_nn(observation)
                loss = torch.mean(weights*(torch.mean(torch.sum((action - predicted_action)** 2, dim=-1), dim=-1)))

                return loss 
            return loss_fn
        else:
            raise NotImplementedError(f"Loss Function for Architecture `{self.policy_cls}` not implemented...")

    def _setup_optimizer(self):
        if self.is_ensemble:
            self.optimizers = [
                self.optimizer_type(self.model.models[i].parameters(), lr=self.lr) for i in range(len(self.model.models))
            ]
        else:
            self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr)

    def _setup_metrics(self):
        if self.is_ensemble:
            self.ensemble_metrics = [defaultdict(list)] * len(self.model.models)
        else:
            self.metrics = defaultdict(list)

    def _reset_model(self):
        if self.is_ensemble:
            num_models = len(self.model.models)
            ensemble_model_type = type(self.model.models[0])
            self.model = None
            print(f"Resetting Ensemble Model...")
            self.model = init_model(ensemble_model_type, self.model_kwargs, device=self.device, num_models=num_models)
        else:
            print(f"Resetting Model...")
            self.model = init_model(self.model_type, self.model_kwargs, device=self.device, num_models=1)

        self.model.to(self.device)

    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, args, auto_only=False, mean=None, std=None):
        #TODO: Add RNN interactive rollout
        data = []
        all_data = []
        if robosuite_cfg is None: 
            low, high = env.action_space
            action_lim = high
        else:
            action_lim = env.action_space.high[0]

        prog_bar = tqdm(range(trajectories_per_rollout), leave=False)
        score = 0
        soft_score = 0


        if len(self.train_demos) > 0 and not auto_only:
            # compute compatibility of training set with model
            train_compatibility = [get_demo_compatibility(self.model, demo, self.device, action_lim=action_lim) for demo in self.train_demos]
            train_likelihood = [np.mean(t[:, 0]) for t in train_compatibility]
            train_entropy = [np.mean(t[:, 1]) for t in train_compatibility]

            if args.sampling_method == 'random':
                # shuffle the demos, likelihood and entropy simultaneously
                zip_demos = list(zip(self.train_demos, train_likelihood, train_entropy))
                random.shuffle(zip_demos)
                self.train_demos, train_likelihood, train_entropy = list(zip(*zip_demos))
            elif args.sampling_method == 'likelihood':
                # sort policy demos by train likelihood
                self.train_demos = [x for _, x in sorted(zip(train_likelihood, self.train_demos), key=lambda pair: pair[0])]
                train_entropy = [x for _, x in sorted(zip(train_likelihood, train_entropy), key=lambda pair: pair[0])]
                train_likelihood = sorted(train_likelihood)

            for n in range(args.teaching_samples):
                visualize_demo(self.train_demos[n], env, args)

        for j in prog_bar:
            rejected = True
            while rejected:
                if auto_only:
                    prog_bar.set_description(f"score: {score}, soft_score: {soft_score}")
                else:
                    prog_bar.set_description(f"Rollout {j+1}/{trajectories_per_rollout}")
                curr_obs, self.expert_mode = env.reset(), False
                if args.web_interface:
                    curr_obs, curr_frame = parse_obs(args, curr_obs)

                init_xml = env.env.sim.model.get_xml()
                init_state = torch.tensor(env.env.sim.get_state().flatten())
                env.env.reset_from_xml_string(init_xml)
                env.env.sim.reset()
                env.env.sim.set_state_from_flattened(init_state)
                env.env.sim.forward()

                done, success = False, False
                success_state = 0
                obs, act, states = [], [], []
                exp_obs, exp_act, exp_states, exp_xml = [], [], [], []
                while not success and not done and len(obs)<robosuite_cfg['max_ep_length']:
                    curr_obs = torch.tensor(curr_obs).float()
                    state = torch.tensor(env.env.sim.get_state().flatten()).float()
                    curr_obs = curr_obs.to(self.device)

 
                    if self.expert_mode and not auto_only:
                        print(len(exp_obs),len(exp_act))
                        # Expert mode (either human or oracle algorithm)

                        a = self.expert_policy.act(curr_obs)
                        if a == "reset":
                            curr_obs = env.reset()
                            init_xml = env.env.sim.model.get_xml()
                            init_state = np.array(env.env.sim.get_state().flatten())
                            env.env.reset_from_xml_string(init_xml)
                            env.env.sim.reset()
                            env.env.sim.set_state_from_flattened(init_state)
                            env.env.sim.forward()
                            obs = []
                            act = []
                            states = []
                            exp_obs, exp_act, exp_states, exp_xml = [], [], [], []
                            done = False
                            success = False
                            self.expert_mode = False
                            print("Switch to Robot")
                            continue
                        a = torch.clamp(a, min=-action_lim, max=action_lim)
                        if self._switch_mode(act=a, obs=curr_obs, action_lim=action_lim):
                            print("Switch to Robot")
                            self.expert_mode = False
                            continue
                        if args.online_feedback:
                            obs.append(curr_obs.cpu())
                            states.append(state.cpu())
                            exp_obs.append(curr_obs.cpu())
                            exp_states.append(state.cpu())
                            exp_xml.append(env.env.sim.model.get_xml())
                            novelty = self.model.get_compatibility(curr_obs.unsqueeze(0), a, type="entropy")
                            lik = self.model.get_compatibility(curr_obs.unsqueeze(0), a, type="likelihood")
                            print(f"Novelty: {novelty.item()}, Likelihood: {lik.item()}")

                        next_obs, success, done, _ = env.step(a.detach())
                        self.env_steps += 1
                        exp_act.append(torch.tensor(a).float().cpu())
                    else:
                        switch_mode = (not auto_only) and self._switch_mode(act=None, obs=None, robosuite_cfg=robosuite_cfg, env=env, action_lim=action_lim)
                        if switch_mode:
                            print("Switch to Expert Mode")
                            self.expert_mode = True
                            self.interact_uncertainty.append(self.model.get_uncertainty(curr_obs, action_lim).cpu())
                            print(f"Uncertainty: {self.interact_uncertainty[-1].item()}")
                            self.num_switches += 1
                            continue
                        obs.append(curr_obs.cpu())
                        states.append(state.cpu())
                        a = self.model.get_action(curr_obs).to(self.device)
                        a = torch.clamp(a, min=-action_lim, max=action_lim)
                        next_obs, success, done, _ = env.step(a.detach())
                        self.env_steps += 1
                        if auto_only and args.environment == 'HammerPlaceEnv':
                            object_pos = env.env.sim.data.body_xpos[env.env.sorting_object_id]
                            object_in_drawer = 1.05 > object_pos[2] > 0.94 and object_pos[1] > 0.00
                            cabinet_closed = env.env.sim.data.qpos[env.env.cabinet_qpos_addrs] > -0.01
                            if cabinet_closed and object_in_drawer and success_state == 2:
                                print("Cabinet closed and object in drawer, success!")
                                success_state = 3
                                soft_score += 0.4
                            elif not cabinet_closed and object_in_drawer and success_state == 1:
                                print("Cabinet open and object in drawer")
                                success_state = 2
                                soft_score += 0.3
                            elif not cabinet_closed and not object_in_drawer and success_state == 0:
                                print("Cabinet open")
                                success_state = 1
                                soft_score += 0.3
                        if success:
                            score += 1
                        
                    act.append(torch.tensor(a).float().cpu())
                    curr_obs = next_obs
                    if args.web_interface:
                        curr_obs, curr_frame = parse_obs(args, curr_obs)
                if not auto_only and not success:
                    rejected = True
                    continue
                print(f"Evaluating recorded_trajectories length: {len(exp_obs)}")
                rejected = False
                if self.num_switches > 0 and len(exp_obs) > 0:
                    # TODO: add rejection specific logs
                    print(f"Expert intervention length: {len(exp_obs)}, {len(exp_act)}")
                    exp_demo = {"obs": exp_obs, "act": exp_act, "states": exp_states, "xml": exp_xml}
                    # TODO: normalize obs option not present
                    comp_matrix = get_demo_compatibility(self.model, exp_demo, self.device, action_lim=action_lim)
                    demo_likelihood = np.mean(comp_matrix[:,0])
                    demo_entropy = np.mean(comp_matrix[:,1])
                    print(f"Demo likelihood: {demo_likelihood} Demo entropy: {demo_entropy}")

                    num_bad = count_bad_samples(comp_matrix, args)

                    print(f"Number of bad samples: {num_bad}/{comp_matrix.shape[0]}")

                    if args.filter_method == 'aggregate':
                        if demo_likelihood > args.likelihood_threshold and demo_entropy < args.entropy_threshold:
                            demo = {"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml}
                            data.append(exp_demo)
                            all_data.append(demo)
                            print(f"Demo accepted: {len(data)}/{args.trajectories_per_rollout} added")
                    elif args.filter_method == 'sample':
                        if num_bad > args.sample_threshold:
                            rejected = True
                        else:
                            data.append(exp_demo)
                            demo = {"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml}
                            all_data.append(demo)
                            self.num_labels += len(exp_demo["obs"])
                            print(f"Labels: {self.num_labels}/{self.max_num_labels} added")
                            print(f"Demo accepted: {len(data)}/{args.trajectories_per_rollout} added")
                            
                    if self.num_labels >= self.max_num_labels:
                        env.close()
                        return data    
                    if rejected:
                        print(f"Demo rejected: {len(data)}/{args.trajectories_per_rollout}")
                        if args.show_bad_samples>0:
                            show_bad_samples(comp_matrix, exp_demo, env, args)
                        if args.show_nearest:
                            nearest_training_demo = get_nearest(exp_demo, self.train_demos) 
                            visualize_demo(nearest_training_demo, env, args)
                    exp_obs, exp_act, exp_states, exp_xml = [], [], [], []
        env.close()
        if args.method == "HGDagger" and not args.web_interface:
            return data, all_data
        return data, success, soft_score

    def _save_checkpoint(self, epoch, best=False, model_idx=0):
        ckpt_name = f"model_{epoch}.pt"
        save_path = os.path.join(self.save_dir, ckpt_name)
        file_exists = os.path.isfile(save_path)
        if self.is_ensemble:
            if not file_exists:
            # Save state dict for each model/optimizer
                ckpt_dict = {
                    "models": [model.state_dict() for model in self.model.models],
                    "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                    "epoch": epoch,
                }
            else:
                ckpt_dict = torch.load(save_path)
                ckpt_dict["models"][model_idx] = self.model.models[model_idx].state_dict()
                ckpt_dict["optimizers"][model_idx] = self.optimizers[model_idx].state_dict()


                
        else:
            ckpt_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}

        if best:
            ckpt_name = f"model_best.pt"
            if self.is_ensemble:
                if len(self.best_ckpt_dict["models"]) > model_idx:
                    self.best_ckpt_dict["models"][model_idx] = ckpt_dict["models"][model_idx] 
                    self.best_ckpt_dict["optimizers"][model_idx] = ckpt_dict["optimizers"][model_idx]
                    self.best_ckpt_dict["epoch"][model_idx] = epoch
                else:
                    self.best_ckpt_dict["models"].append(ckpt_dict["models"][model_idx]) 
                    self.best_ckpt_dict["optimizers"].append(ckpt_dict["optimizers"][model_idx])
                    self.best_ckpt_dict["epoch"].append(epoch)
            else:
                self.best_ckpt_dict = ckpt_dict
            save_path = os.path.join(self.save_dir, ckpt_name)
            torch.save(self.best_ckpt_dict, save_path)
        else:
            ckpt_name = f"model_{epoch}.pt"
            save_path = os.path.join(self.save_dir, ckpt_name)
            torch.save(ckpt_dict, save_path)

    def _save_metrics(self):
        if self.is_ensemble:
            for i, metrics in enumerate(self.ensemble_metrics):
                save_path = os.path.join(self.save_dir, f"model{i}_metrics.pkl")
                df = pd.DataFrame(metrics)
                df.to_pickle(save_path)
        else:
            save_path = os.path.join(self.save_dir, "metrics.pkl")
            df = pd.DataFrame(self.metrics)
            df.to_pickle(save_path)

    def _update_metrics(self, **kwargs):
        if self.is_ensemble:
            for metrics in self.ensemble_metrics:
                for key, val in kwargs.items():
                    metrics[key].append(val)
        else:
            for key, val in kwargs.items():
                self.metrics[key].append(val)

    def train(self, model, optimizer, train_loader, val_loader, args, model_idx=0):
        model.train()
        # torch.autograd.set_detect_anomaly(True)
        best_epoch = 0
        best_val_loss = float("inf")
        for epoch in range(args.epochs):
            prog_bar = tqdm(train_loader, leave=False)
            prog_bar.set_description(f"Epoch {epoch}/{args.epochs - 1}")
            epoch_losses = []
            for (obs, act, w) in prog_bar:
                optimizer.zero_grad()
                obs, act, w = obs.to(self.device), act.to(self.device), w.to(self.device)
                # Custom Loss Function per Model Architecture
                obs = obs.float()
                w = w.float()
                act = act.float()
                loss = self.loss_fn(model, observation=obs, action=act, weights=w)
                loss.backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                epoch_losses.append(loss.item())
                prog_bar.set_postfix(train_loss=loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_loss = self.validate(model, val_loader, args)

            print(f"Epoch {epoch} Train Loss: {avg_loss}")
            print(f"Epoch {epoch} Val Loss: {avg_val_loss}")

            # Update metrics
            self._update_metrics(epoch=epoch, train_loss=avg_loss, val_loss=avg_val_loss)

            if epoch % args.save_iter == 0 or epoch == args.epochs - 1:
                self._save_checkpoint(epoch, model_idx=model_idx)

            if avg_val_loss < best_val_loss:
                self._save_checkpoint(epoch, best=True, model_idx=model_idx)
                best_val_loss = avg_val_loss
                best_epoch = epoch

            wandb.log(
                {"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val_loss}
            ) 
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
 
    def validate(self, model, val_loader, args):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (obs, act, w) in val_loader:
                obs, act, w = obs.to(self.device), act.to(self.device), w.to(self.device)
                obs = obs.float()
                w = w.float()
                act = act.float()
                # Custom Loss Function per Model Architecture
                loss = self.loss_fn(model, observation=obs, action=act, weights=w)
                val_losses.append(loss.item())
        model.train()
        return sum(val_losses) / len(val_losses)

    def run(self, train_loader, val_loader, args, env=None, robosuite_cfg=None, mean=None, std=None) -> None:
        raise NotImplementedError

    def _save_eval_data(self, data):
        save_path = os.path.join(self.save_dir, f"data_eval.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def eval_rollout(self, env, robosuite_cfg, trajectories_per_rollout, args, mean=None, std=None):
        # TODO: Add termination on success
        n_envs = env.num_envs
        episode_rewards = []
        episode_soft_score = []
        episode_lengths = []
        episode_counts = np.zeros(n_envs, dtype="int")
        if self.policy_cls == "RNN":
            D = 2 if args.bidirectional else 1
            if not self.is_ensemble:
                hidden_state = (torch.zeros(n_envs, D*args.num_layers, args.hidden_size).to(self.device),
                                torch.zeros(n_envs, D*args.num_layers, args.hidden_size).to(self.device))
            else:
                hidden_state = (torch.zeros(args.num_models, n_envs, D*args.num_layers, args.hidden_size).to(self.device),
                                torch.zeros(args.num_models, n_envs, D*args.num_layers, args.hidden_size).to(self.device)) 

        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(trajectories_per_rollout + i) // n_envs for i in range(n_envs)], dtype="int")
        current_rewards = np.zeros(n_envs)
        current_soft_score = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        env_demos = []
        env_obs, env_act, env_success = [[]*n_envs] * 3
        if robosuite_cfg is None: 
            low, high = env.action_space
            action_lim = high
        else:
            action_lim = env.action_space.high[0]

        curr_obs = env.reset()
        start = time.time()
        while (episode_counts < episode_count_targets).any():
            curr_obs = torch.tensor(curr_obs).float()
            # TODO: store eval rollouts 
            # for i in range(n_envs):
            #     env_obs[i].append(curr_obs[i].cpu())
            curr_obs = curr_obs.to(self.device)
            
            if self.policy_cls == "RNN":
                # make hidden state contiguous
                hidden_state = (hidden_state[0].contiguous(), hidden_state[1].contiguous())
                a, hidden_state = self.model.get_action(curr_obs, state=hidden_state)
                a = a.squeeze(1)
            else:
                a = self.model.get_action(curr_obs).to(self.device)
            a = torch.clamp(a, min=-action_lim, max=action_lim)
            next_obs, success, done, _ = env.step(a.detach())
            if args.environment == "HammerPlaceEnv":
                current_soft_score = np.array(success)
            current_lengths += 1

            curr_obs = next_obs
            if args.environment == "SawyerCoffee":
                for i in range (len(success)):
                    current_rewards[i] += success[i]['task']
            else:
                current_rewards += success
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    if done[i]:
                        if args.environment == "HammerPlaceEnv":
                            if current_soft_score[i] == 1.0:
                                episode_rewards.append(1)
                            else:
                                episode_rewards.append(0)
                        else:
                            if current_rewards[i] > 0:
                                episode_rewards.append(1)
                            else:
                                episode_rewards.append(0)
                        print(f"Current soft score: {current_soft_score[i]} total soft score: {sum(episode_soft_score)}")
                        episode_soft_score.append(current_soft_score[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        current_soft_score[i] = 0
                        if self.policy_cls == "RNN":
                            if not self.is_ensemble:
                                hidden_state[0][i, :, :] = (torch.zeros(D*args.num_layers, args.hidden_size).to(self.device))
                                hidden_state[1][i, :, :] = (torch.zeros(D*args.num_layers, args.hidden_size).to(self.device))
                            else:
                                hidden_state[0][ :, i, :, :] = (torch.zeros(args.num_models, D*args.num_layers, args.hidden_size).to(self.device))
                                hidden_state[1][ :, i, :, :] = (torch.zeros(args.num_models, D*args.num_layers, args.hidden_size).to(self.device))
                        wandb.run.summary["success_rate"] = np.mean(episode_rewards)    
                        wandb.run.summary["soft_score"] = np.mean(episode_soft_score)
                        print(f"{current_rewards[i]} Current Reward: {episode_rewards[-1]} Episode Rewards: {sum(episode_rewards)} Episodes: {episode_counts.sum()} Iter Time: {(time.time() - start)/episode_counts.sum()}")
        return episode_rewards, episode_soft_score, env_demos
    
    def eval_auto(self, args, env=None, robosuite_cfg=None, mean=None, std=None):
        data, success, soft_score = self._rollout(env, robosuite_cfg, args.N_eval_trajectories, args, auto_only=True)
        save_file = os.path.join(self.save_dir, "eval_auto_data.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(data, f)
        # wandb.run.summary["success_rate"] = sum(successes) / len(successes)
        # print(f"Success rate: {sum(successes)/len(successes)}")
        # print(f"Eval data saved to {save_file}")

    def eval_multi_proc(self, args, robosuite_cfg=None, mean=None, std=None):
        max_traj_len = robosuite_cfg['max_ep_length'] if robosuite_cfg is not None else None
        def get_env(idx):
            def _f():
                new_env, _ = setup_robosuite(args, max_traj_len=max_traj_len)
                new_env.env.seed(idx)
                return new_env
            return _f
        args.no_render = True
        eval_env = SubprocVecEnv([get_env(i) for i in range(args.n_procs)], start_method='spawn')
        episode_rewards, episode_lengths, eval_demos = self.eval_rollout(eval_env, robosuite_cfg, args.N_eval_trajectories, args, mean=mean, std=std)
        # self._save_eval_data(eval_demos)
        wandb.run.summary["success_rate"] = np.mean(episode_rewards)    
        wandb.run.summary["mean_episode_length"] = np.mean(episode_lengths)
        print(f"Success rate: {np.mean(episode_rewards)}")