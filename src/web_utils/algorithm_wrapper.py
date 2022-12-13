import random

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from util.int_utils import *
from algos import BaseAlgorithm
from web_utils.vis_utils import *
from datasets.util import get_policy_data

class WebWrapper(BaseAlgorithm):
    def __init__(self, algorithm, status):
        self.algorithm = algorithm    
        self.device = algorithm.device
        self.run = self.algorithm.run
        # override algorithm _rollout method with web version
        self.algorithm._rollout = self._rollout
        self.model = self.algorithm.model
        self.eval_multi_proc = self.algorithm.eval_multi_proc
        self.web_status = status
        self.dagger_stage = 0
    
    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, args, auto_only=False, mean=None, std=None):
        
        print("Rollout...")
        data = []
        save_path = os.path.join(self.algorithm.save_dir, f"data_epoch{self.dagger_stage}.pkl")
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            print(f"Resuming demo collection from epoch {len(data)}")
 
        action_lim = env.action_space.high[0]

        prog_bar = tqdm(range(trajectories_per_rollout), leave=False)
        score = 0

        if len(self.algorithm.train_demos) > 0 and not auto_only:
            # compute compatibility of training set with model
            train_compatibility = [get_demo_compatibility(self.algorithm.model, demo, self.device, action_lim=action_lim) for demo in self.algorithm.train_demos]
            train_likelihood = [np.mean(t[:, 0]) for t in train_compatibility]
            train_entropy = [np.mean(t[:, 1]) for t in train_compatibility]

            if args.sampling_method == 'random':
                # shuffle the demos, likelihood and entropy simultaneously
                zip_demos = list(zip(self.algorithm.train_demos, train_likelihood, train_entropy))
                random.shuffle(zip_demos)
                self.algorithm.train_demos, train_likelihood, train_entropy = list(zip(*zip_demos))
            elif args.sampling_method == 'likelihood':
                # sort policy demos by train likelihood
                self.algorithm.train_demos = [x for _, x in sorted(zip(train_likelihood, self.algorithm.train_demos), key=lambda pair: pair[0])]
                train_entropy = [x for _, x in sorted(zip(train_likelihood, train_entropy), key=lambda pair: pair[0])]
                train_likelihood = sorted(train_likelihood)

        # only show from the top 80% of the demos
        demos_to_show = self.algorithm.train_demos[:int(0.8*len(self.algorithm.train_demos))]
        random.shuffle(demos_to_show)
        if len(data) == 0:
            for i in range(args.teaching_samples):
                self.web_status["state"] = f"<h3> Learn {i+1}/{args.teaching_samples} </h3> Here are {args.teaching_samples} demonstrations that the robot learned from. Try and give similar feedback to the robot."
                print(f"Teaching sample {i}")
                web_visualize_demo(self.web_status, env, demos_to_show[i], args)
        count = 0
        for j in prog_bar:
            if j != len(data):
                continue
            rejected = True
            while rejected:
                if auto_only:
                    prog_bar.set_description(f"score: {score} {score/j:.2f}")
                else:
                    prog_bar.set_description(f"Rollout {j}/{trajectories_per_rollout}")


                curr_obs, self.algorithm.expert_mode = env.reset(), False
                init_xml = env.env.sim.model.get_xml()
                init_state = torch.tensor(env.env.sim.get_state().flatten())
                env.env.reset_from_xml_string(init_xml)
                env.env.sim.reset()
                env.env.sim.set_state_from_flattened(init_state)
                env.env.sim.forward()
                curr_obs, curr_frame = parse_obs(args, curr_obs)
                img = env.env.sim.render(
                    camera_name="agentview",
                    width=args.img_w,
                    height=args.img_h,
                    depth=False,
                )
                self.web_status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
                curr_obs = web_settle_actions(self.web_status, env, args, reset=True)
                curr_obs, curr_frame = parse_obs(args, curr_obs)
                done, success = False, False
                obs, act, states = [], [], []
                exp_obs, exp_act, exp_states, exp_xml = [], [], [], []
                while not success and not done and len(obs)<robosuite_cfg['max_ep_length']:
                    self.web_status["state"] = f"<h3>Intervene {j+1}/{trajectories_per_rollout} Steps: {len(obs)}/{robosuite_cfg['max_ep_length']}</h3> <br>The robot is trying to complete the task. Intervene by pressing 'z' and give feedback to the robot when you think it is stuck or is taking an incorrect action. The robot will reset if it can't complete the task in {robosuite_cfg['max_ep_length']} steps."
                    curr_obs = torch.tensor(curr_obs).float()
                    state = torch.tensor(env.env.sim.get_state().flatten()).float()
                    curr_obs = curr_obs.to(self.device)
                    if self.algorithm.expert_mode and not auto_only:
                        # Expert mode (either human or oracle algorithm)
                        a = self.algorithm.expert_policy.act(curr_obs)
                        if a == "reset":
                            curr_obs = env.reset()
                            init_xml = env.env.sim.model.get_xml()
                            init_state = np.array(env.env.sim.get_state().flatten())
                            env.env.reset_from_xml_string(init_xml)
                            env.env.sim.reset()
                            env.env.sim.set_state_from_flattened(init_state)
                            env.env.sim.forward()
                            curr_obs = web_settle_actions(self.web_status, env, args, reset=True)
                            curr_obs, curr_frame = parse_obs(args, curr_obs)
                            obs = []
                            act = []
                            states = []
                            done = False
                            success = False
                            self.algorithm.expert_mode = False
                            print("Switch to Robot")
                            self.web_status["compatibility"] = .5
                            continue
                        a = torch.clamp(a, min=-action_lim, max=action_lim)
                        if self.algorithm._switch_mode(act=a, obs=curr_obs, action_lim=action_lim):
                            print("Switch to Robot")
                            self.web_status["compatibility"] = .5
                            self.algorithm.expert_mode = False
                            continue
                        elif args.online_feedback:
                            novelty = self.algorithm.model.get_compatibility(curr_obs.unsqueeze(0), a, type="entropy").item()
                            lik = self.algorithm.model.get_compatibility(curr_obs.unsqueeze(0), a, type="likelihood").item()
                            if novelty > args.entropy_threshold:
                                comp_scalar = 0.0
                            if novelty < args.entropy_threshold:
                                comp_scalar = min((lik/args.likelihood_threshold), 1)
                            # comp_scalar = min(max(lik-args.likelihood_threshold/4, 0) * max(args.entropy_threshold/2-novelty, 0) / (args.likelihood_threshold/2 * args.entropy_threshold/2),1)
                            self.web_status["compatibility"] = 1-comp_scalar
                            print(f"Novelty: {novelty}, Likelihood: {lik}, Compatibility: {comp_scalar}")
                        states.append(state.cpu())
                        obs.append(curr_obs.cpu())
                        exp_obs.append(curr_obs.cpu())
                        exp_states.append(state.cpu())
                        exp_xml.append(env.env.sim.model.get_xml())
                        next_obs, success, done, _ = env.step(a.detach())
                        self.algorithm.env_steps += 1
                        exp_act.append(torch.tensor(a).float().cpu())
                        act.append(torch.tensor(a).float().cpu())
                    else:
                        exp_switch_act = self.algorithm._switch_mode(act=None, obs=None, robosuite_cfg=robosuite_cfg, env=env, action_lim=action_lim)
                        if exp_switch_act == "reset":
                            curr_obs = env.reset()
                            init_xml = env.env.sim.model.get_xml()
                            init_state = np.array(env.env.sim.get_state().flatten())
                            env.env.reset_from_xml_string(init_xml)
                            env.env.sim.reset()
                            env.env.sim.set_state_from_flattened(init_state)
                            env.env.sim.forward()
                            curr_obs = web_settle_actions(self.web_status, env, args, reset=True)
                            curr_obs, curr_frame = parse_obs(args, curr_obs)
                            obs = []
                            act = []
                            states = []
                            done = False
                            success = False
                            self.algorithm.expert_mode = False
                            print("Switch to Robot")
                            self.web_status["compatibility"] = .5
                            continue    
                        switch_mode = (not auto_only) and exp_switch_act
                        if switch_mode:
                            print("Switch to Expert Mode")
                            self.algorithm.expert_mode = True
                            self.algorithm.interact_uncertainty.append(self.algorithm.model.get_uncertainty(curr_obs, action_lim).cpu())
                            self.algorithm.num_switches += 1
                            continue
                        states.append(state.cpu())
                        obs.append(curr_obs.cpu())
                        a = self.algorithm.model.get_action(curr_obs).to(self.device)
                        a = torch.clamp(a, min=-action_lim, max=action_lim)
                        next_obs, success, done, _ = env.step(a.detach())
                        self.algorithm.env_steps += 1
                        if success:
                            score += 1
                        act.append(torch.tensor(a).float().cpu())
                    curr_obs = next_obs
                    img = env.env.sim.render(
                        camera_name="agentview",
                        width=args.img_w,
                        height=args.img_h,
                        depth=False,
                    )
                    self.web_status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
                    curr_obs, success, done, _ = web_settle_actions(self.web_status, env, args, a)
                    curr_obs, curr_frame = parse_obs(args, curr_obs)
                print("obsactlen", len(obs), len(act))
                if not auto_only and not success:
                    rejected = True
                    continue
                print(count)    
                print(f"Evaluating recorded_trajectories length: {len(exp_obs)}")
                rejected = False
                if self.algorithm.num_switches > 0 and len(exp_obs) > 0 and not auto_only:
                    # TODO: add rejection specific logs
                    print(f"Expert intervention length: {len(exp_obs)}, {len(exp_act)}")
                    exp_demo = {"obs": exp_obs, "act": exp_act, "states": exp_states, "xml": exp_xml}
                    comp_matrix = get_demo_compatibility(self.algorithm.model, exp_demo, self.device, action_lim=action_lim)
                    demo_likelihood = np.mean(comp_matrix[:,0])
                    demo_entropy = np.mean(comp_matrix[:,1])
                    print(f"Demo likelihood: {demo_likelihood} Demo entropy: {demo_entropy}")

                    num_bad = count_bad_samples(comp_matrix, args)

                    print(f"Number of bad samples: {num_bad}/{comp_matrix.shape[0]}")

                    if args.filter_method == 'aggregate':
                        if demo_likelihood > args.likelihood_threshold and demo_entropy < args.entropy_threshold:
                            demo = {"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml}
                            data.append(exp_demo)
                            print(f"Demo accepted: {len(data)}/{args.trajectories_per_rollout} added")
                    elif args.filter_method == 'sample':
                        if num_bad > args.sample_threshold:
                            rejected = True
                        else:
                            data.append(exp_demo)
                            with open(save_path, "wb") as f:
                                pickle.dump(data, f)

                    if rejected:
                        print(f"Demo rejected: {len(data)}/{args.trajectories_per_rollout}")
                        if args.show_bad_samples>0:
                            web_show_bad_samples(self.web_status, comp_matrix, exp_demo, env, args)
                        if args.show_nearest:
                            self.web_status["state"] = f"<h3>Here is a demonstration that the robot learned form. Try and give similar feedback.</h3>"
                            nearest_training_demo = get_nearest(exp_demo, self.algorithm.train_demos) 
                            web_visualize_demo(self.web_status, env, nearest_training_demo, args)
                    exp_obs, exp_act, exp_states = [], [], []

        print(score)
        self.dagger_stage += 1
        self.web_status["state"] = f"<h3>You have completed {self.dagger_stage}/{args.dagger_epochs} stages. Thank you for you feedback.  The robot is training again and will ask for more feedback in about 2 minutes!"
        return data

class DaggerWebWrapper(BaseAlgorithm):
    def __init__(self, algorithm, status):
        self.algorithm = algorithm    
        self.device = algorithm.device
        # override algorithm _rollout method with web version
        self.algorithm._rollout = self._rollout
        self._save_data = self.algorithm._save_data
        self.model = self.algorithm.model
        self.num_labels, self.max_num_labels = self.algorithm.num_labels, self.algorithm.max_num_labels
        self.eval_multi_proc = self.algorithm.eval_multi_proc
        self.web_status = status
        self.dagger_stage = 0

    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, args, auto_only=False):
        # check if the demo collection had been resumed
        save_path = os.path.join(self.algorithm.save_dir, f"data_epoch{self.dagger_stage}.pkl")
        data = []
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            print(f"Resuming demo collection from epoch {len(data)}")
        print("Rollout...")
        action_lim = env.action_space.high[0]

        prog_bar = tqdm(range(trajectories_per_rollout), leave=False)
        score = 0

        if len(data) == 0 and args.online_feedback:
            if len(self.algorithm.train_demos) > 0 and not auto_only:
                # compute compatibility of training set with model
                train_compatibility = [get_demo_compatibility(self.algorithm.model, demo, self.device, action_lim=action_lim) for demo in self.algorithm.train_demos]
                train_likelihood = [np.mean(t[:, 0]) for t in train_compatibility]
                train_entropy = [np.mean(t[:, 1]) for t in train_compatibility]

                if args.sampling_method == 'random':
                    # shuffle the demos, likelihood and entropy simultaneously
                    zip_demos = list(zip(self.algorithm.train_demos, train_likelihood, train_entropy))
                    random.shuffle(zip_demos)
                    self.algorithm.train_demos, train_likelihood, train_entropy = list(zip(*zip_demos))
                elif args.sampling_method == 'likelihood':
                    # sort policy demos by train likelihood
                    self.algorithm.train_demos = [x for _, x in sorted(zip(train_likelihood, self.algorithm.train_demos), key=lambda pair: pair[0])]
                    train_entropy = [x for _, x in sorted(zip(train_likelihood, train_entropy), key=lambda pair: pair[0])]
                    train_likelihood = sorted(train_likelihood)

            # only show from the top 80% of the demos
            demos_to_show = self.algorithm.train_demos[:int(0.8*len(self.algorithm.train_demos))]
            random.shuffle(demos_to_show)
            for i in range(args.teaching_samples):
                self.web_status["state"] = f"<h3> Learn {i+1}/{args.teaching_samples} </h3> Here are {args.teaching_samples} demonstrations that the robot learned from. Try and give similar feedback to the robot."
                print(f"Teaching sample {i}")
                web_visualize_demo(self.web_status, env, demos_to_show[i], args)
 
        count = 0
        for j in prog_bar:
            if j != len(data):
                continue
            rejected = True
            while rejected:
                if auto_only:
                    prog_bar.set_description(f"score: {score} {score/j:.2f}")
                else:
                    prog_bar.set_description(f"Rollout {j}/{trajectories_per_rollout}")


                curr_obs, self.algorithm.expert_mode = env.reset(), False
                init_xml = env.env.sim.model.get_xml()
                init_state = torch.tensor(env.env.sim.get_state().flatten())
                env.env.reset_from_xml_string(init_xml)
                env.env.sim.reset()
                env.env.sim.set_state_from_flattened(init_state)
                env.env.sim.forward()
                curr_obs, curr_frame = parse_obs(args, curr_obs)
                img = env.env.sim.render(
                    camera_name="agentview",
                    width=args.img_w,
                    height=args.img_h,
                    depth=False,
                )
                self.web_status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
                curr_obs = web_settle_actions(self.web_status, env, args, reset=True)
                curr_obs, curr_frame = parse_obs(args, curr_obs)
                done, success = False, False
                obs, act, states = [], [], []
                exp_obs, exp_act, exp_states = [], [], [] 
                while not success and not done and len(obs)<robosuite_cfg['max_ep_length']:
                    self.web_status["state"] = f"<h3>Teach {j}/{trajectories_per_rollout} Steps: {len(obs)}/{robosuite_cfg['max_ep_length']}</h3> <br>Show the robot how to complete the task. Try and complete the task in ways that are similar to the robots thinking. The robot will reset if it can't complete the task in {robosuite_cfg['max_ep_length']} steps. Press 'q' to reset the robot if you think that the robot should not learn from this demonstration."
                    curr_obs = torch.tensor(curr_obs).float()
                    state = torch.tensor(env.env.sim.get_state().flatten()).float()
                    curr_obs = curr_obs.to(self.device)
                    if not auto_only:
                        # Expert mode (either human or oracle algorithm)
                        a = self.algorithm.expert_policy.act(curr_obs)
                        if a == "reset":
                            curr_obs = env.reset()
                            init_xml = env.env.sim.model.get_xml()
                            init_state = np.array(env.env.sim.get_state().flatten())
                            env.env.reset_from_xml_string(init_xml)
                            env.env.sim.reset()
                            env.env.sim.set_state_from_flattened(init_state)
                            env.env.sim.forward()
                            robosuite_cfg['input_device'].start_control()
                            curr_obs = web_settle_actions(self.web_status, env, args, reset=True)
                            curr_obs, curr_frame = parse_obs(args, curr_obs)
                            obs = []
                            act = []
                            states = []
                            exp_obs = []
                            exp_states = []
                            exp_act = []
                            done = False
                            success = False
                            continue
                        a = torch.clamp(a, min=-action_lim, max=action_lim)
                        if args.online_feedback:

                            novelty = self.algorithm.model.get_compatibility(curr_obs.unsqueeze(0), a, type="entropy").item()
                            lik = self.algorithm.model.get_compatibility(curr_obs.unsqueeze(0), a, type="likelihood").item()
                            comp_scalar = 0.
                            if novelty > args.entropy_threshold:
                                comp_scalar = 0.0
                            if novelty <= args.entropy_threshold:
                                comp_scalar = min((lik/args.likelihood_threshold), 1)
                                # comp_scalar = min(max(lik-args.likelihood_threshold/4, 0)/(1-args.likelihood_threshold/4), 1)
 
                            # comp_scalar = min(max(lik-args.likelihood_threshold/4, 0) * max(args.entropy_threshold/2-novelty, 0) / (args.likelihood_threshold/2 * args.entropy_threshold/2),1)
                            self.web_status["compatibility"] = 1-comp_scalar
                            print(f"Novelty: {novelty}, Likelihood: {lik}, Compatibility: {comp_scalar}")
                        states.append(state.cpu())
                        obs.append(curr_obs.cpu())
                        exp_obs.append(curr_obs.cpu())
                        exp_states.append(state.cpu())
                        next_obs, success, done, _ = env.step(a.detach())
                        self.algorithm.env_steps += 1
                        exp_act.append(torch.tensor(a).float().cpu())
                    else:
                        raise NotImplementedError
                    act.append(torch.tensor(a).float().cpu())
                    curr_obs = next_obs
                    img = env.env.sim.render(
                        camera_name="agentview",
                        width=args.img_w,
                        height=args.img_h,
                        depth=False,
                    )
                    self.web_status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
                    curr_obs, success, done, _ = web_settle_actions(self.web_status, env, args, a)
                    curr_obs, curr_frame = parse_obs(args, curr_obs)
                print("obsactlen", len(obs), len(act))
                if not auto_only and not success:
                    rejected = True
                    continue
                print(count)    
                rejected = False
                if len(exp_obs) > 0 and not auto_only:
                    print(f"Expert intervention length: {len(exp_obs)}, {len(exp_act)}")
                    exp_demo = {"obs": exp_obs, "act": exp_act, "states": exp_states, "init_xml": init_xml}
                    comp_matrix = get_demo_compatibility(self.algorithm.model, exp_demo, self.device, action_lim=action_lim)
                    demo_likelihood = np.mean(comp_matrix[:,0])
                    demo_entropy = np.mean(comp_matrix[:,1])
                    print(f"Demo likelihood: {demo_likelihood} Demo entropy: {demo_entropy}")
                    num_bad = count_bad_samples(comp_matrix, args)
                    print(f"Number of bad samples: {num_bad}/{comp_matrix.shape[0]}")
                    if args.filter_method == 'aggregate':
                        if demo_likelihood > args.likelihood_threshold and demo_entropy < args.entropy_threshold:
                            demo = {"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml}
                            data.append(exp_demo)
                            with open(save_path, "wb") as f:
                                pickle.dump(data, f)
                            print(f"Demo accepted: {len(data)}/{args.trajectories_per_rollout} added")
                    elif args.filter_method == 'sample':
                        if num_bad > args.sample_threshold:
                            rejected = True
                        else:
                            data.append(exp_demo)
                            with open(save_path, "wb") as f:
                                pickle.dump(data, f)
                    if rejected:
                        print(f"Demo rejected: {len(data)}/{args.trajectories_per_rollout}")
                        if args.show_bad_samples>0:
                            web_show_bad_samples(self.web_status, comp_matrix, exp_demo, env, args)
                        if args.show_nearest:
                            self.web_status["state"] = f"<h3>Here is a demonstration that the robot learned form. Try and give similar feedback.</h3>"
                            nearest_training_demo = get_nearest(exp_demo, self.algorithm.train_demos) 
                            web_visualize_demo(self.web_status, env, nearest_training_demo, args)
                    exp_obs, exp_act, exp_states = [], [], []
        self.dagger_stage += 1
        if args.online_feedback:
            self.web_status["state"] = f"<h3>You have completed {self.dagger_stage}/{args.dagger_epochs} stages. Thank you for your demonstrations.  The robot is training again and will ask for more demonstrations in about 2 minutes!"
        return data
        
    def run(self, train_data, val_data, args, env, robosuite_cfg) -> None:
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        # initial policy training + dagger interactive training
        new_data = []
        with open(args.data_path, "rb") as f:
            self.algorithm.train_demos = pickle.load(f)

        dagger_epochs = args.dagger_epochs if args.init_dagger_model else args.dagger_epochs + 1

        for epoch in range(dagger_epochs):
            ckpt = None
            self.dagger_stage = epoch
            if args.robosuite:
                robosuite_cfg["input_device"].start_control()
            if args.init_dagger_model and epoch == 0:
                ckpt = torch.load(args.model_path, map_location=self.device)
            # elif args.best_epoch and epoch > 0:
            #     model_path = os.path.join(self.algorithm.save_dir, "model_best.pt")
            #     ckpt = torch.load(model_path, map_location=self.device)
            if ckpt is not None:
                if args.num_models > 1:
                    for ensemble_model, state_dict in zip(self.model.models, ckpt["models"]):
                        ensemble_model.load_state_dict(state_dict)
                else:
                    self.model.load_state_dict(ckpt["model"])
            # Roll out trajectories
            if epoch > 0 or args.init_dagger_model:
                #TODO: add collection of expert demonstrations 
                new_data = self._rollout(env, robosuite_cfg, args.trajectories_per_rollout, args)
            # Add new data to training dataset only after first iteration, as we want to train on the initial dataset
            self._save_data(new_data, epoch)
            for demos in new_data:
                for obs, act in zip(demos["obs"], demos["act"]):
                    train_data.update_buffer(obs, act)
            if self.num_labels > self.max_num_labels:
                break

            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            # Reset the model and optimizer for retraining
            if epoch > 0:
                self.algorithm._reset_model()
                self.algorithm._setup_optimizer()

            # Retrain policy
            if args.online_feedback and not args.init_dagger_model:
                if self.algorithm.is_ensemble:
                    for i, (model, optimizer) in enumerate(zip(self.model.models, self.algorithm.optimizers)):
                        self.algorithm.train(model, optimizer, train_loader, val_loader, args, model_idx=i)
                else:
                    self.algorithm.train(self.model, self.algorithm.optimizer, train_loader, val_loader, args)

            # if self.use_indicator_beta:
            #     # Set beta to 0 after first interaction
            #     if epoch == 1:
            #         self.beta = 0
            # else:
            #     # Beta decays exponentially
            #     self.beta *= self.beta