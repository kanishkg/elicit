import random

import torch

from constants import REACH2D_ACT_MAGNITUDE, REACH2D_ACT_DIM, REACH2D_OBS_DIM, REACH2D_PILLAR_OBS_DIM, REACH2D_SUCCESS_THRESH


class Pillar:
    """
    Class defining a rectangular "Pillar" obstacle, either centered at the midpoint of
    'start' and 'goal' if fixed_loc == None, otherwise centered at fixed_loc.
    """

    def __init__(self, height, width, start, goal, device, fixed_loc=None, overlap_padding=0.0) -> None:
        # Set dimensions
        self.height = height
        self.width = width
        self.device = device

        # Set location (self.x, self.y = upper left corner coordinates)
        # Pillar is centered at the midpoint of start and goal if
        # fixed_loc == None, else centered at fixed_loc.
        self.fixed_loc = fixed_loc
        self.x, self.y = self._init_loc(start, goal)
        self.loc = torch.tensor([self.x, self.y], device=device)

        # Padding to use when checking if a state overlaps with
        # the pillar
        self.overlap_padding = overlap_padding

    def _init_loc(self, start, goal):
        if self.fixed_loc == None:
            center = (start + goal) / 2
            x = center[0] - self.width / 2
            y = center[1] + self.height / 2
        else:
            center = self.fixed_loc
            x = center[0] - self.width / 2
            y = center[1] + self.height / 2

        return x, y

    def overlaps(self, state):
        """
        Returns whether or not a state overlaps the pillar.
        """
        state_x, state_y = state
        x_overlaps = state_x >= self.x and state_x <= (self.x + self.width)
        y_overlaps = state_y <= self.y and state_y >= (self.y - self.height)

        return x_overlaps and y_overlaps

    def _padded_overlaps(self, state):
        """
        Returns whether or not a state overlaps the pillar with extra padding;
        this is a more relaxed verison of overlaps().
        """
        state_x, state_y = state
        x_overlaps = state_x >= (self.x + self.overlap_padding) and state_x <= (
            self.x + self.width - self.overlap_padding
        )
        y_overlaps = state_y <= (self.y - self.overlap_padding) and state_y >= (
            self.y - self.height + self.overlap_padding
        )

        return x_overlaps and y_overlaps


class Reach2DPillar:
    def __init__(
        self,
        device,
        max_ep_len,
        grid=None,
        pillar_height=1.0,
        pillar_width=1.0,
        fixed_loc=None,
        overlap_padding=0.1,
        random_start_state=False,
        range_x=3.0,
        range_y=3.0,
    ):
        self.device = device
        self.max_ep_len = max_ep_len
        self.random_start_state = random_start_state
        -REACH2D_ACT_MAGNITUDE, REACH2D_ACT_MAGNITUDE
        self.action_dim = REACH2D_ACT_DIM 
        self.action_spec = (-REACH2D_ACT_MAGNITUDE,
                            REACH2D_ACT_MAGNITUDE)
        # TODO - Figure what is the right attribute name to use here.
        self.action_space = self.action_spec

        self.grid = grid
        self.pillar_height = pillar_height
        self.pillar_width = pillar_width
        self.fixed_loc = fixed_loc
        self.overlap_padding = overlap_padding

        self.range_x = range_x
        self.range_y = range_y

        self.curr_state, self.goal_state, self.pillar = self._init_obs()

        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.act_dim = REACH2D_ACT_DIM
        self.obs_dim = REACH2D_PILLAR_OBS_DIM
        self.ep_len = 0

    def _init_obs(self):
        valid = False
        while not valid:
            self.curr_state = self._init_start_state()
            self.goal_state = self._init_goal_state()

            self.pillar = Pillar(
                self.pillar_height,
                self.pillar_width,
                self.curr_state,
                self.goal_state,
                self.device,
                fixed_loc=self.fixed_loc,
                overlap_padding=self.overlap_padding,
            )

            valid = not (self.pillar.overlaps(self.curr_state) or self.pillar.overlaps(self.goal_state))
        return self.curr_state, self.goal_state, self.pillar

    def _init_start_state(self):
        if self.random_start_state:
            if self.grid == None:
                start_state = torch.tensor(
                    [random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device
                )
            else:
                start_state = self.grid.sample_random_state()

        else:
            start_state = torch.zeros(2, device=self.device)

        return start_state

    def _init_goal_state(self):
        if self.grid == None:
            goal_state = torch.tensor(
                [random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device
            )
        else:
            goal_state = self.grid.sample_random_state()

        return goal_state

    def close(self):
        """
        Empty function so callers don't break with use of this class.
        """
        pass

    def reset(self):
        self.curr_state, self.goal_state, self.pillar = self._init_obs()
        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.ep_len = 0
        return self.curr_obs

    def step(self, action):
        self.curr_state += action
        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.ep_len += 1
        return self.curr_obs, self._check_success(), self._check_done(), None

    def _check_success(self):
        if self.grid == None:
            return (torch.norm(self.curr_state - self.goal_state) <= REACH2D_SUCCESS_THRESH).item()
        else:
            return torch.isclose(torch.norm(self.curr_state - self.goal_state), torch.zeros(1), atol=1e-5).item()

    def _check_done(self):
        return self.ep_len > self.max_ep_len or self.pillar._padded_overlaps(self.curr_state)
