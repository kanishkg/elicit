import glfw
import gym
import numpy as np

from constants import ACTION_BATCH_SIZE


class WebCustomWrapper(gym.Env):
    def __init__(self, env, render, simplify_actions=True, settle_actions=True):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self._render = render
        self.simplify_actions = simplify_actions
        self.settle_actions = settle_actions

    def reset(self):
        glfw.terminate()
        if not glfw.init():
            glfw.init()
        r = self.env.reset()
        self.render()
        if self.settle_actions:
            settle_action = np.zeros(self.action_space.shape[0])
                # settle_action[-1] = -1
            for _ in range(10):
                r, r2, r3, r4 = self.env.step(settle_action)
                self.render()
            self.gripper_closed = False
        return r

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action = action.cpu().numpy()
        action_ = action.copy()
        if self.simplify_actions:
            action_[3] = 0.0
            action_[4] = 0.0
        r1, r2, r3, r4 = self.env.step(action_)
        self.render()
        settle_action = np.zeros(self.action_space.shape[0])
        settle_action[-1] = action[-1]
        if self.settle_actions:
            for _ in range(ACTION_BATCH_SIZE):
                r1, r2, r3, r4 = self.env.step(settle_action)
                self.render()
            if self.env.env.robot_configs[0]['gripper_type'] != "WipingGripper":
                if action[-1] > 0:
                    self.gripper_closed = True
                else:
                    self.gripper_closed = False

        return r1, self._check_success(), r3, r4

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()