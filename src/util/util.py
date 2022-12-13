from statistics import mode
from unicodedata import bidirectional
from gym import Space
import time

import usb.core
import torch
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.devices import Keyboard

from robosuite.wrappers import GymWrapper 

try:
    import robosuite_task_zoo
except ImportError:
    print("Could not import robosuite_task_zoo. Please install robosuite_task_zoo to use extra envs.")
try:
    from robosuite.wrappers import VisualizationWrapper
except ImportError:
    print("Could not import robosuite.wrappers.VisualizationWrapper")

from constants import HAMMER_MAX_TRAJ_LEN, TRANSPORT_MAX_TRAJ_LEN, DOOR_MAX_TRAJ_LEN, WIPE_MAX_TRAJ_LEN, ACTION_BATCH_SIZE, NUT_ASSEMBLY_MAX_TRAJ_LEN, PICKPLACE_MAX_TRAJ_LEN, REACH2D_ACT_MAGNITUDE, NUT_ASSEMBLY_SQUARE_MAX_TRAJ_LEN, CAN_MAX_TRAJ_LEN, COFFEE_MAX_TRAJ_LEN
from envs import CustomWrapper
from models import MDN, MLP, Ensemble, GaussianMLP, LinearModel, qEnsemble, RNN
from web_utils.controller import WebKeyboard
from web_utils.webenv_wrapper import WebCustomWrapper
import threading

#TODO: Move these configs to a different file
low_dim_keys_coffee = ['eef_pos', 'eef_quat', 'object-state', 'gripper_qpos']

low_dim_keys = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object-state",
        ]
low_dim_keys_hammer = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object-state",
            "joint_success_qpos"
        ]
low_dim_keys_two_arm = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "robot1_eef_pos", 
            "robot1_eef_quat", 
            "robot1_gripper_qpos", 
            "object-state",
        ]
keys_to_include = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin',
                  'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat',
                  'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object-state']
WIPE_CONFIG = {
    # settings for reward
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
    "wipe_contact_reward": 0.01,  # reward for contacting something with the wiping tool
    "unit_wiped_reward": 50.0,  # reward per peg wiped
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold
    "distance_multiplier": 5.0,  # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
    "distance_th_multiplier": 5.0,  # multiplier in the tanh function for the aforementioned reward
    # settings for table top
    "table_full_size": [0.5, 0.8, 0.05],  # Size of tabletop
    "table_offset": [0.15, 0, 0.9],  # Offset of table (z dimension defines max height of table)
    "table_friction": [0.03, 0.005, 0.0001],  # Friction parameters for the table
    "table_friction_std": 0,  # Standard deviation to sample different friction parameters for the table each episode
    "table_height": 0.0,  # Additional height of the table over the default location
    "table_height_std": 0.0,  # Standard deviation to sample different heigths of the table each episode
    "line_width": 0.04,  # Width of the line to wipe (diameter of the pegs)
    "two_clusters": False,  # if the dirt to wipe is one continuous line or two
    "coverage_factor": 0.6,  # how much of the table surface we cover
    "num_markers": 100,  # How many particles of dirt to generate in the environment
    # settings for thresholds
    "contact_threshold": 1.0,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold": 0.5,  # force threshold (N) to overcome to get increased contact wiping reward
    "pressure_threshold_max": 60.0,  # maximum force allowed (N)
    # misc settings
    "print_results": False,  # Whether to print results or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": False,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
}



def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x

def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


try:
    from robosuite.devices import SpaceMouse
    class BetterSpaceMouse(SpaceMouse):
        def __init__(self, vendor_id=9583, product_id=50735, pos_sensitivity=1.0, rot_sensitivity=1.0):

            print("Opening SpaceMouse device")
            self.device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
            ep=self.device[0].interfaces()[0].endpoints()[0]
            i=self.device[0].interfaces()[0].bInterfaceNumber
            self.device.reset()
            if self.device.is_kernel_driver_active(i):
                print("Detaching kernel driver")
                self.device.detach_kernel_driver(i)
            self.device.set_configuration()
            self.eaddr = ep.bEndpointAddress
            self.pos_sensitivity = pos_sensitivity
            self.rot_sensitivity = rot_sensitivity


            # 6-DOF variables
            self.x, self.y, self.z = 0, 0, 0
            self.roll, self.pitch, self.yaw = 0, 0, 0

            self._display_controls()

            self.single_click_and_hold = False

            self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self._reset_state = 0
            self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            self._enabled = True

            # launch a new listener thread to listen to SpaceMouse
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()

        def run(self):
            """Listener method that keeps pulling new messages."""

            t_last_click = -1

            while True:
                d = self.device.read(self.eaddr, 13, 10000000)
                if d is not None and self._enabled:

                    if d[0] == 1:  ## readings from 6-DoF sensor
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0

                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])

                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]

                    elif d[0] == 3:  ## readings from the side buttons

                        # press left button
                        if d[1] == 1:
                            t_click = time.time()
                            elapsed_time = t_click - t_last_click
                            t_last_click = t_click
                            self.single_click_and_hold = True

                        # release left button
                        if d[1] == 0:
                            self.single_click_and_hold = False

                        # right button is for reset
                        if d[1] == 2:
                            self._reset_state = 1
                            self._enabled = True
                            self._reset_internal_state()
except ImportError:
    class BetterSpaceMouse(object):
        def __init__(self, vendor_id=9583, product_id=50735, pos_sensitivity=1.0, rot_sensitivity=1.0):
            raise NotImplementedError("SpaceMouse robosuite implementation not found. Update robosuite")


def get_compatibility(model, policy, device, action_lim=None):
    likelihood_comp = []
    entropy_comp = []
    for demo in policy:
        obs = torch.stack(demo["obs"], dim=0).to(device)
        act = torch.stack(demo["act"], dim=0).to(device)
        likelihood_comp.append(model.get_compatibility(obs, act, type="likelihood", action_lim=action_lim, reduction="obs"))
        entropy_comp.append(model.get_compatibility(obs, act, type="entropy", action_lim=action_lim, reduction="obs"))
    likelihood_comp = torch.cat(likelihood_comp, dim=0).cpu().numpy()
    entropy_comp = torch.cat(entropy_comp, dim=0).cpu().numpy()
    # obs x 2 (likelihood, entropy)
    compatibility_matrix = np.stack([likelihood_comp, entropy_comp], axis=1)
    print(compatibility_matrix.shape)
    return compatibility_matrix


def get_model_type_and_kwargs(args, obs_dim, act_dim):
    if args.arch == "LinearModel":
        model_type = LinearModel
        if args.environment == "Reach2D":
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, scale=REACH2D_ACT_MAGNITUDE, normalize=True)
        else:
            model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim)
    elif args.arch == "MLP":
        model_type = MLP
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size,
                            layer_norm=args.layer_norm, dropout=args.dropout)
    elif args.arch == "GaussianMLP":
        model_type = GaussianMLP
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size)
    elif args.arch == "MDN":
        model_type = MDN
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size)
    elif args.arch == "RNN":
        model_type = RNN
        model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout)
    else:
        raise NotImplementedError(f"The architecture {args.arch} has not been implemented yet!")

    return model_type, model_kwargs


def init_model(model_type, model_kwargs, device, num_models):
    if num_models > 1:
        model_kwargs = dict(model_kwargs=model_kwargs, device=device, num_models=num_models, model_type=model_type)
        model = Ensemble(**model_kwargs)
    elif num_models == 1:
        model = model_type(**model_kwargs)
    else:
        raise ValueError(f"Got {num_models} for args.num_models, but value must be an integer >= 1!")

    return model

def init_q_model(obs_dim, act_dim, hidden_size, num_models, device):
    model = qEnsemble(obs_dim, act_dim, hidden_size, num_models, device)
    return model

def setup_robosuite(args, max_traj_len):
    render = not args.no_render
    has_offscreen = False
    use_camera_obs = False
    if args.web_interface:
        render = False
        has_offscreen = True
        use_camera_obs = True
    if "Sawyer" not in args.environment:
        controller_config = load_controller_config(default_controller="OSC_POSE")
        config = {
            "env_name": args.environment,
            "robots": "UR5e",
            "controller_configs": controller_config,
        }

    if args.environment == "NutAssembly":
        if args.nut_type == "round":
            env = suite.make(
                **config,
                has_renderer=render,
                has_offscreen_renderer=has_offscreen,
                render_camera="agentview",
                single_object_mode=2,  # env has 1 nut instead of 2
                nut_type="round",
                ignore_done=False,
                use_camera_obs=use_camera_obs,
                reward_shaping=False,
                control_freq=20,
                hard_reset=True,
                use_object_obs=True,
                horizon=(
                    ACTION_BATCH_SIZE + NUT_ASSEMBLY_MAX_TRAJ_LEN * (ACTION_BATCH_SIZE + 1)
                ),  # TODO: clean this-- this is a result of thriftydagger code setup
            )
        elif args.nut_type == "square":
            config = {
            "env_name": args.environment,
            "robots": "Panda",
            "controller_configs": controller_config,
            }
            env = suite.make(
                **config,
                has_renderer=render,
                has_offscreen_renderer=has_offscreen,
                render_camera="agentview",
                single_object_mode=2,  # env has 1 nut instead of 2
                nut_type="square",
                ignore_done=False,
                use_camera_obs=use_camera_obs,
                reward_shaping=False,
                control_freq=20,
                hard_reset=True,
                use_object_obs=True,
                horizon=(
                    NUT_ASSEMBLY_SQUARE_MAX_TRAJ_LEN
                ),  # TODO: clean this-- this is a result of thriftydagger code setup
            )

    elif args.environment == "PickPlaceCan":
        config = {
            "env_name": args.environment,
            "robots": "Panda",
            "controller_configs": controller_config,
            }

        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=False,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            horizon=(
                CAN_MAX_TRAJ_LEN
            ),  # TODO: clean this-- this is a result of thriftydagger code setup
        ) 

    elif args.environment == "SawyerCoffee":
        controller_name = 'EE_POS_ORI' 
        controller_config = load_controller_config(default_controller=controller_name)
        config = {
            "env_name": "SawyerCoffeeContactTeleop",
            "controller_config": controller_config,
            }

        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            camera_name="agentview",
            ignore_done=False,
            use_camera_obs=False,
            reward_shaping=False,
            control_freq=40,
            use_object_obs=True,
        )

    elif args.environment == "TwoArmTransport":
        config = {
            "env_name": args.environment,
            "robots": ["Panda", "Panda"],
            "controller_configs": controller_config,
            }
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=False,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            horizon=(
                TRANSPORT_MAX_TRAJ_LEN
            ),  # TODO: clean this-- this is a result of thriftydagger code setup
        ) 


    elif args.environment == "Wipe":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=has_offscreen,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=use_camera_obs,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            task_config=WIPE_CONFIG,
            horizon=(
                ACTION_BATCH_SIZE+ WIPE_MAX_TRAJ_LEN * (ACTION_BATCH_SIZE + 1)
            ),  # TODO: clean this-- this is a result of thriftydagger code setup
        )

    elif args.environment == "Door":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=has_offscreen,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=use_camera_obs,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            horizon=(
               DOOR_MAX_TRAJ_LEN 
            ),  # TODO: clean this-- this is a result of thriftydagger code setup
        )

    elif args.environment == "HammerPlaceEnv":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=has_offscreen,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=use_camera_obs,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            horizon=ACTION_BATCH_SIZE+ HAMMER_MAX_TRAJ_LEN * (ACTION_BATCH_SIZE + 1),
        )

    elif args.environment == "PickPlace":
        env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2,
            object_type="cereal",
            ignore_done=False,
            use_camera_obs=False,
            reward_shaping=False,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True,
            horizon=ACTION_BATCH_SIZE+ PICKPLACE_MAX_TRAJ_LEN * (ACTION_BATCH_SIZE + 1),
        )


    else:
        raise NotImplementedError(args.environment)

    if args.environment == "HammerPlaceEnv":
        if args.low_dim:
            env = GymWrapper(env, keys=low_dim_keys) 
        else:
            env = GymWrapper(env)
    elif args.environment == "SawyerCoffee":
        if args.low_dim:
            env = GymWrapper(env, keys=low_dim_keys_coffee) 
        else:
            raise NotImplementedError("high_dim not implemented for SawyerCoffee")
    elif args.nut_type == "square" and args.environment == "NutAssembly":
        obs = env.reset()
        if args.low_dim:
            modality_dims = {key: obs[key].shape for key in low_dim_keys}
            print(modality_dims)
            print(env.robots)
            env = GymWrapper(env, keys=low_dim_keys)   
        else:
            env = GymWrapper(env, keys=keys_to_include)
    elif args.low_dim:
        if "TwoArm" in args.environment:
            obs = env.reset()
            modality_dims = {key: obs[key].shape for key in low_dim_keys}
            print(modality_dims)
            env = GymWrapper(env, keys=low_dim_keys_two_arm)
        else:
            env = GymWrapper(env, keys=low_dim_keys)
    else:
        env = GymWrapper(env)
    if "SawyerCoffee" not in args.environment:
        env = VisualizationWrapper(env, indicator_configs=None)
    if args.web_interface:
        custom_wrapper = WebCustomWrapper 
    else:
        custom_wrapper = CustomWrapper 
    if "SawyerCoffee" in args.environment:
        env = custom_wrapper(env, render=render, simplify_actions=False, settle_actions=False)
    elif args.nut_type == "square" or args.space_mouse or args.environment in ["Door", "Can"]:
        env = custom_wrapper(env, render=render, simplify_actions=False, settle_actions=False)
    elif args.environment not in ["NutAssembly", "Wipe", "HammerPlaceEnv", "PickPlace"]:
        env = custom_wrapper(env, render=render, simplify_actions=False, settle_actions=True)
    elif args.web_interface:
        env = custom_wrapper(env, render=render, simplify_actions=True, settle_actions=False)
    else:
        print("settle_actions=True")
        env = custom_wrapper(env, render=render, simplify_actions=True, settle_actions=True)

    if args.space_mouse:
        input_device = BetterSpaceMouse(vendor_id=9583, product_id=50770)
    elif not args.web_interface:
        input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)
    else:
        input_device = WebKeyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)

    if render and not args.space_mouse:
        env.viewer.add_keypress_callback("any", input_device.on_press)
        env.viewer.add_keyup_callback("any", input_device.on_release)
        env.viewer.add_keyrepeat_callback("any", input_device.on_press)

    if 'Sawyer' in args.environment:
        arm_ = "right"
        config_ = "single-arm-opposed"
        # active_robot = env.robots[arm_ == "left"]
        robosuite_cfg = {
            "max_ep_length": max_traj_len,
            "input_device": input_device,
            "arm": arm_,
            "env_config": config_,
            # "active_robot": active_robot,
        }
    else:
        arm_ = "right"
        config_ = "single-arm-opposed"
        active_robot = env.robots[arm_ == "left"]
        robosuite_cfg = {
            "max_ep_length": max_traj_len,
            "input_device": input_device,
            "arm": arm_,
            "env_config": config_,
            "active_robot": active_robot,
        }
    print(f"obs dim {env.observation_space.shape[0]}")
    return env, robosuite_cfg
