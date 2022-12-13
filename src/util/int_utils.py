import argparse
import numpy as np
import os
import pickle
import random
import time

import torch
try:
    from dtaidistance import dtw_ndim
except ValueError:
    print("Please install dtaidistance to use this script")

from constants import (
    HAMMER_MAX_TRAJ_LEN,
    NUT_ASSEMBLY_MAX_TRAJ_LEN,
    PICKPLACE_MAX_TRAJ_LEN,
    WIPE_MAX_TRAJ_LEN,
    DOOR_MAX_TRAJ_LEN,
    CAN_MAX_TRAJ_LEN,
    TRANSPORT_MAX_TRAJ_LEN,
    COFFEE_MAX_TRAJ_LEN,
    REACH2D_ACT_MAGNITUDE,
    REACH2D_MAX_TRAJ_LEN,
    REACH2D_PILLAR_MAX_TRAJ_LEN,
    REACH2D_RANGE_X,
    REACH2D_RANGE_Y,
    ACTION_BATCH_SIZE
)

from util import get_model_type_and_kwargs, init_model, setup_robosuite
from robosuite.wrappers import GymWrapper 
try:
    from robosuite.wrappers import VisualizationWrapper
except ImportError:
    print("Could not import robosuite.wrappers.VisualizationWrapper")
import robosuite as suite
from envs import CustomWrapper
from robosuite import load_controller_config
from robosuite.utils.mjcf_utils import postprocess_model_xml


# compatibility map for old keys (MLP without drop or layer norm)
backward_key_map = {
   'layers.4.weight' : 'layers.8.weight', 
   'layers.4.bias' : 'layers.8.bias',
   'layers.2.weight': 'layers.4.weight',
   'layers.2.bias': 'layers.4.bias',
}

def load_model(args, obs_dim, act_dim, device):
    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim, act_dim)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
    model.to(device)
    model.eval()
    # print model state dict keys
    ckpt = torch.load(args.model_path, map_location=device)

    if args.num_models > 1:
        for ensemble_model, state_dict in zip(model.models, ckpt["models"]):
            if args.backward:
                for old_key, new_key in backward_key_map.items():
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)
            ensemble_model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt["model"])
    return model



def get_demo_compatibility(model, demo, device, action_lim=None):
    if type(demo["obs"]) is list:
        obs = torch.stack(demo["obs"]).to(device)
        act = torch.stack(demo["act"]).to(device)
    else:
        obs = demo["obs"].to(device) 
        act = demo["act"].to(device)
    likelihood_comp = model.get_compatibility(obs, act, type="likelihood", action_lim=action_lim, reduction="obs")
    entropy_comp = model.get_compatibility(obs, act, type="entropy", action_lim=action_lim, reduction="obs")
    # obs x 2 (likelihood, entropy)
    compatibility_matrix = np.stack([likelihood_comp.cpu().detach(), entropy_comp.cpu().detach()], axis=1)
    return compatibility_matrix

def parse_obs(args, obs):
    if args.environment == "NutAssembly":
        if args.nut_type == "round":
            curr_obs = np.concatenate((obs[:14], obs[-37:]))
            curr_frame = np.reshape(obs[14:-37], (256, 256, 3))/255.
        elif args.nut_type == "square":
            raise NotImplementedError("Square nut not implemented yet!")
    elif args.environment == "Wipe":
        curr_obs = np.concatenate((obs[:700], obs[-26:]))
        curr_frame = np.reshape(obs[700:-26], (256, 256, 3))/255.
    elif args.environment == "HammerPlaceEnv":
        curr_obs = np.concatenate((obs[:29], obs[-45:]))
        curr_frame = np.reshape(obs[29:-45], (256, 256, 3))/255.
    else:
        raise NotImplementedError(f"{args.environment}  has not been implemented!")
    return curr_obs, curr_frame

def get_env(args):
    if args.robosuite:
        if args.environment == "PickPlace":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=PICKPLACE_MAX_TRAJ_LEN)
        elif args.environment == "NutAssembly":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_MAX_TRAJ_LEN)
        elif args.environment == "HammerPlaceEnv":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=HAMMER_MAX_TRAJ_LEN)
        elif args.environment == "PickPlaceCan":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=CAN_MAX_TRAJ_LEN)
        elif args.environment == "SawyerCoffee":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=COFFEE_MAX_TRAJ_LEN)
        elif args.environment == "TwoArmTransport":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=TRANSPORT_MAX_TRAJ_LEN)
        elif args.environment == "Wipe":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=WIPE_MAX_TRAJ_LEN)
        elif args.environment == "Door":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=DOOR_MAX_TRAJ_LEN)
        else:
            raise NotImplementedError(f"Environment {args.environment} has not been implemented yet!")
    else:
        raise NotImplementedError(f"Only Robosuite has been implemented!")
    return env, robosuite_cfg

def visualize_demo(demo, env, args):
    env.reset()
    # TODO: Partial trajectories cant use actions, and there is no way to visualize 
    # TODO: partial trajectories correctly for Wipe
    # if 'init_xml' in demo and args.environment == "Wipe":
    #     # initializing using xml does not work with robomimic data
    #     # as it is collected from the offline_data branch of robosuite
    #     model_xml = demo['init_xml']
    #     xml = postprocess_model_xml(model_xml)
    #     env.env.reset_from_xml_string(xml)
    # env.env.sim.reset()
    env.env.sim.set_state_from_flattened(demo["states"][0])
    env.env.sim.forward()
    for step in range(len(demo["obs"])):
        obs = demo["obs"][step]
        state = demo["states"][step]
        act = demo["act"][step]
        if args.use_actions:
            if "Sawyer" in args.environment:
                act = act.cpu().numpy()
            env.step(act)
            env.render()
            time.sleep(0.3)
        else:
            # using state will not erase the markings in Wipe
            env.env.sim.set_state_from_flattened(state)
            env.env.sim.forward()

            obs = env.env._get_observations()
            env.env.render()
            time.sleep(0.5)

def store_demo(demo, env, keys_filter):
    #TODO these observations are not correct
    env.reset()
    env.env.sim.set_state_from_flattened(demo["states"][0])
    env.env.sim.forward()
    new_obs = []
    for step in range(len(demo["obs"])):
        obs = demo["obs"][step]
        state = demo["states"][step]
        act = demo["act"][step]
        # using state will not erase the markings in Wipe
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        print(obs)
        obs = env.env._get_observations()
        ob_lst = []
        for k in keys_filter:
            ob_lst.append(np.array(obs[k]).flatten())
        no = np.concatenate(ob_lst)
        print(no)
        new_obs.append(torch.tensor(no).float())
        env.render()
    assert len(new_obs) == len(demo["obs"])
    return new_obs



def sample_trajectory(env, policy):
    """ 
    Samples a demonstration interactively from the environment with input from the user. 
    We aren't using the robosuite DataCollectionWrapper because it doesn't support interactive rejection.
    """
    print("Press 'Q' to reset (and ignore) the current demonstration and ctrl-c to quit and save the demos.")
    action_lim = env.action_space.high[0]
    env.reset()
    env.render()

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
    done = False
    success = False
    while not done and not success:
        action = policy.act(curr_obs)
        # Use 'Q' press to indicate 'reset'
        if action == "reset":
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
            done = False
            success = False
            continue
        action = torch.clamp(action, min=-action_lim, max=action_lim)
        obs.append(torch.tensor(curr_obs).float())
        act.append(action.float())
        states.append(torch.tensor(env.env.sim.get_state().flatten()).float())
        curr_obs, success, done, _ = env.step(action)
    demo = {"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml}
    return demo 

def get_nearest(demo, policy_demos):
    """
    Finds the nearest demonstration in policy_demos to the user demo.
    We use dynamic time warping to find the nearest demonstration.
    Distance is between the observations.
    """
    min_dist, min_idx = float("inf"), -1
    for i, policy_demo in enumerate(policy_demos):
        obs_policy = np.array([o.numpy() for o in policy_demo["obs"]]) 
        obs_demo = np.array([o.numpy() for o in demo["obs"]]) 
        try:
            dist = dtw_ndim.distance(obs_policy, obs_demo)
        except:
            dist = 0
        if dist < min_dist:
            min_dist, min_idx = dist, i
    return policy_demos[min_idx]

def count_bad_samples(comp_matrix, args):
    num_bad = 0
    for sample in range(comp_matrix.shape[0]):
        if comp_matrix[sample, 0] > args.likelihood_threshold and comp_matrix[sample, 1] < args.entropy_threshold:
            num_bad += 1
    return num_bad

def count_good_samples(comp_matrix, args):
    num_good = 0
    for sample in range(comp_matrix.shape[0]):
        if comp_matrix[sample, 0] < args.likelihood_threshold/2 and comp_matrix[sample, 1] > args.entropy_threshold*1.5:
            num_good += 1
    return num_good

def show_bad_samples(comp_matrix, demo, env, args):
    likelihood = []
    entropy = []
    step = []
    for sample in range(comp_matrix.shape[0]):
        if comp_matrix[sample, 0] > args.likelihood_threshold and comp_matrix[sample, 1] < args.entropy_threshold:
            likelihood.append(comp_matrix[sample, 0])
            entropy.append(comp_matrix[sample, 1])
            step.append(sample)
    # sort step in descending order of likelihood
    step = [s for _, s in sorted(zip(likelihood, step), reverse=True)] 
    entropy = [e for _, e in sorted(zip(likelihood, entropy), reverse=True)] 
    likelihood = sorted(likelihood, reverse=True) 
    for i in range(min(args.show_bad_samples, len(step))):
        low = max(0, step[i] - args.window_size)
        high = min(len(demo["obs"]), step[i] + args.window_size)
        sub_demo = {"obs": demo["obs"][low:high],
                    "act": demo["act"][low:high],
                    "states": demo["states"][low:high],
                    "init_xml": demo["xml"][low]}
        print(f"Bad sample {i+1}/{len(step)} with likelihood {likelihood[i]}, entropy {entropy[i]}")
        visualize_demo(sub_demo, env, args)
    