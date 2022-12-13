import glfw
import gym
import numpy as np

from constants import ACTION_BATCH_SIZE


class CustomWrapper(gym.Env):
    def __init__(self, env, render, simplify_actions=True, settle_actions=True):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.success_state = 0
        self.soft_score = 0
        try:
            self.robots = env.robots
        except AttributeError:
            print("working with old env")
        self._render = render
        self.simplify_actions = simplify_actions
        self.settle_actions = settle_actions

    def reset(self):
        r = self.env.reset()
        self.render()
        self.success_state = 0
        self.soft_score = 0
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

        # TODO: hacky way to check if env is hammer
        if self.observation_space.shape[0] == 74:
            object_pos = self.env.sim.data.body_xpos[self.env.sorting_object_id]
            object_in_drawer = 1.05 > object_pos[2] > 0.94 and object_pos[1] > 0.00
            cabinet_closed = self.env.sim.data.qpos[self.env.cabinet_qpos_addrs] > -0.01
            if cabinet_closed and object_in_drawer and self.success_state == 2:
                print("success")
                self.success_state = 3
                self.soft_score += 0.4
            elif not cabinet_closed and object_in_drawer and self.success_state == 1:
                print("hamemr in drawer")
                self.success_state = 2
                self.soft_score += 0.3
            elif not cabinet_closed and not object_in_drawer and self.success_state == 0:
                print("drawer open")
                self.success_state = 1
                self.soft_score += 0.3
            return r1, self.soft_score, r3, r4
        return r1, self._check_success(), r3, r4 

    def _check_success(self):
        success = self.env._check_success()
        return success

    def render(self):
        if self._render:
            self.env.render()
