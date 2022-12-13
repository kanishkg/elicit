"""
franka_env.py

Core abstraction over the physical Robot hardware, sensors, and internal robot state. Follows a standard
OpenAI Gym-like API.
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch
from core.perception.camera import Camera
from gym import Env
from polymetis import GripperInterface, RobotInterface


# Silence OpenAI Gym Warnings
gym.logger.setLevel(logging.ERROR)


# fmt: off
HOMES = {
    "default": ([0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],),

    # Task "sunny side"
    "sunny-side": ([-0.9818, 0.1361, 0.3580, -1.6153, 1.4822, 1.1530, 0.6945],
                   [-0.7817, 0.8116, 0.2610, -1.6978, 1.5946, 1.8677, -0.1078]),

    # >> Home Positions for the RB2 Tasks... ignore!
    # "pour": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    # "scoop": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    # "zip": [-0.1337, 0.3634, -0.1395, -2.3153, 0.1478, 2.7733, -1.1784],
    # "insertion": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
}
# fmt: on

# Control Frequency & other useful constants...
#   > Ref: Gripper constants from: https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
HZ, GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 30, 0.3, 120, 0.08570, 0.01

# Joint Controller gains -- we want a compliant robot when recording, and stiff when playing back / operating
KQ_GAINS = {
    "record": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "rb2": [26.6667, 40.0000, 33.3333, 33.3333, 23.3333, 16.6667, 6.6667],
    "default": [80, 120, 100, 100, 70, 50, 20],
    "stiff": [240.0, 360.0, 300.0, 300.0, 210.0, 150.0, 60.0],
}
KQD_GAINS = {
    "record": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "rb2": [3.3333, 3.3333, 3.3333, 3.3333, 1.6667, 1.6667, 1.6667],
    "default": [10, 10, 10, 10, 5, 5, 5],
    "stiff": [30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0],
}

# End-Effector Controller gains -- we want a compliant robot when recording, and stiff when playing back / operating
KX_GAINS = {
    "record": [1, 1, 1, 1, 1, 1],
    "default": [150, 150, 150, 10, 10, 10],
    "stiff": [450, 450, 450, 30, 30, 30],  # Upper bound is ???, so using joint default -- 3x multiplier
}
KXD_GAINS = {
    "record": [1, 1, 1, 1, 1, 1],
    "default": [25, 25, 25, 7, 7, 7],
    "stiff": [75, 75, 75, 21, 21, 21],  # Upper bound is ???, so using joint default -- 3x multiplier
}


# Hardcoded Low/High Joint Thresholds for the Franka Emika Panda Arm
LOW_JOINTS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH_JOINTS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


class FrankaEnv(Env):
    def __init__(
        self, home: str, hz: int, mode: str, controller: str = "joint", camera: bool = True, grasp: bool = False
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified in joint space - 7-DoF for Pandas)
        :param hz: Default policy control Hz; somewhere between 20-60 is a good range.
        :param mode: Mode in < "record" | "playback" | "control"> -- mostly used to set gains!
        :param controller: Which impedance controller to use in < joint | cartesian > (demonstrating always uses joint)
        :param camera: Whether or not to log camera observations (RGB)
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.camera = Camera() if camera else None
        self.gripper, self.grasp = None, grasp

        # Other Constants to be Initialized...
        self.robot, self.kp, self.kpd = None, None, None
        self.current_joint_pose, self.current_ee_pose, self.initial_ee_pose = None, None, None

        # Initialize Robot and PD Controller
        self.reset()

    def robot_setup(self, do_grasp: bool = False, franka_ip: str = "172.16.0.1") -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = RobotInterface(ip_address=franka_ip)

        # Home Loop --> if "multiple" homes, we're specifying a staged reset...
        homes = HOMES.get(self.home, "default")
        for home_pos in homes:
            self.robot.set_home_pose(torch.Tensor(home_pos))
            self.robot.go_home()
            time.sleep(1)

        # Initialize current joint & EE poses...
        self.current_joint_pose = self.robot.get_joint_positions().numpy()
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.initial_ee_pose = self.current_ee_pose

        # Create an *Impedance Controller*, with the desired gains...
        #   > Note: Feel free to add any other controller, e.g., a PD controller around joint poses.
        #           =>> Ref: https://github.com/AGI-Labs/franka_control/blob/master/util.py#L74
        if self.controller == "joint":
            self.robot.start_joint_impedance(Kq=self.kp, Kqd=self.kpd)
        elif self.controller == "cartesian":
            self.robot.start_cartesian_impedance(Kx=self.kp, Kxd=self.kpd)
        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def reset(self, do_grasp: bool = False) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup(do_grasp=do_grasp)
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])

        # Note that deltas are "shifted" 1 time step to the right from the corresponding "state"
        obs = {
            "q": new_joint_pose,
            "qdot": self.robot.get_joint_velocities().numpy(),
            "delta_q": new_joint_pose - self.current_joint_pose,
            "ee_pose": new_ee_pose,
            "delta_ee_pose": new_ee_pose - self.current_ee_pose,
        }
        if self.camera:
            obs["rgb"] = self.camera.get_frame()

        # Bump "current" poses...
        self.current_joint_pose, self.current_ee_pose = new_joint_pose, new_ee_pose
        return obs

    def step(self, action: Optional[np.ndarray]) -> Tuple[Dict[str, np.ndarray], int, bool, None]:
        """Run a step in the environment, where `delta` specifies if we are sending absolute poses or deltas in poses!"""
        if action is not None:
            if self.controller == "joint":
                self.robot.update_desired_joint_positions(torch.from_numpy(action))
            elif self.controller == "cartesian":
                # First 3 elements are xyz, last 4 elements are quaternion orientation...
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(action[:3]), orientation=torch.from_numpy(action[3:])
                )

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), 0, False, None

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented for Physical FrankaEnv...")

    def close(self) -> Any:
        # Terminate Policy
        logs = self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None
        time.sleep(1)

        return logs
