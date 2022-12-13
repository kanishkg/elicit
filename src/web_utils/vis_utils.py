from PIL import Image
import time

import numpy as np
import torch

from constants import *
from util.util import *
from util.int_utils import *
from policies import *

def web_settle_actions(status, env, args, action=None, reset=False):
    settle_action = np.zeros(env.action_space.shape[0])
    if not reset:
        settle_action[-1] = action[-1]
    for _ in range(ACTION_BATCH_SIZE):
        obs, r2, r3, r4 = env.env.step(settle_action)
        curr_obs, curr_frame = parse_obs(args, obs)
        img = env.env.sim.render(
            camera_name="agentview",
            width=args.img_w,
            height=args.img_h,
            depth=False,
        )
        status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
    if reset:
        env.gripper_closed = False
        return obs
    if action[-1] > 0:
        env.gripper_closed = True
    else:
        env.gripper_closed = False
    return obs, env._check_success(), r3, r4

def web_visualize_demo(status, env, demo, args):
    env.reset()
    if 'init_xml' in demo:
        # initializing using xml does not work with robomimic data
        # as it is collected from the offline_data branch of robosuite
        model_xml = demo['init_xml']
        xml = postprocess_model_xml(model_xml)
        env.env.reset_from_xml_string(xml)
    env.env.sim.reset()
    print()
    env.env.sim.set_state_from_flattened(demo["states"][0])
    env.env.sim.forward()
    if args.use_actions:
        web_settle_actions(status, env, args, reset=True)
    for step in range(len(demo["obs"])):
        obs = demo["obs"][step]
        state = demo["states"][step]
        act = demo["act"][step]
        if args.use_actions:
            obs, _,_,_ = env.step(act)
            img = env.env.sim.render(
                camera_name="agentview",
                width=args.img_w,
                height=args.img_h,
                depth=False,
            )
            status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
            web_settle_actions(status, env, args, action=act)


        else:
            env.env.sim.set_state_from_flattened(state)
            env.env.sim.forward()
            img = env.env.sim.render(
                camera_name="agentview",
                width=args.img_w,
                height=args.img_h,
                depth=False,
            )
            status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
            time.sleep(0.3)
           
def web_show_bad_samples(status, comp_matrix, demo, env, args):
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
        status["state"] = f"<h3> Rejected, here is where you went wrong {i}/{args.show_bad_samples} </h3>"
        low = max(0, step[i] - args.window_size)
        high = min(len(demo["obs"]), step[i] + args.window_size)
        sub_demo = {"obs": demo["obs"][low:high],
                    "act": demo["act"][low:high],
                    "states": demo["states"][low:high]}
                    
        print(f"Bad sample {i+1}/{len(step)} with likelihood {likelihood[i]}, entropy {entropy[i]}")
        web_visualize_demo(status, env, sub_demo, args)
 
def web_collect_demo(status, env, robosuite_cfg, N_trajectories, args):
    policy = NutAssemblyPolicy(args.policy, env, robosuite_cfg)
    action_lim = env.action_space.high[0]
    env.reset()
    demos = []
    while len(demos) < N_trajectories:
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
        print(f"recorded {len(demos)}/{N_trajectories} demos")
        curr_obs, curr_frame = parse_obs(args, curr_obs)
        img = env.env.sim.render(
            camera_name="agentview",
            width=args.img_w,
            height=args.img_h,
            depth=False,
        )
        status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
        curr_obs = web_settle_actions(status, env, args, reset=True)
        curr_obs, curr_frame = parse_obs(args, curr_obs)
        while not done and not success:
            status["state"] = f"<h3> Practice: {len(demos)}/{args.N_initial} done &emsp; Steps {len(obs)}/{robosuite_cfg['max_ep_length']} </h3> <br>Practice the task and complete it {args.N_initial} times. The robot will reset if the task is not completed in {robosuite_cfg['max_ep_length']} steps"
            action = policy.act(curr_obs)
            if action == "reset":
                curr_obs = env.reset()
                init_xml = env.env.sim.model.get_xml()
                init_state = np.array(env.env.sim.get_state().flatten())
                env.env.reset_from_xml_string(init_xml)
                env.env.sim.reset()
                env.env.sim.set_state_from_flattened(init_state)
                env.env.sim.forward()
                curr_obs = web_settle_actions(status, env, args, reset=True)
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
            curr_obs, curr_frame = parse_obs(args, curr_obs)
            img = env.env.sim.render(
            camera_name="agentview",
            width=args.img_w,
            height=args.img_h,
            depth=False,
            )
            status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
            curr_obs, success, done, _ = web_settle_actions(status, env, args, action)
            curr_obs, curr_frame = parse_obs(args, curr_obs)
        if not success:
            continue
        demos.append({"obs": obs, "act": act, "success": success, "states": states, "init_xml": init_xml})
    status["state"] = f"<h3> Practice: {len(demos)}/{args.N_initial} done" 
    return demos