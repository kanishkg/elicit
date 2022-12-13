import torch

from constants import REACH2D_ACT_DIM, REACH2D_ACT_MAGNITUDE


class Reach2DPolicy:
    def __init__(self, policy, model=None) -> None:
        self.policy = policy
        self.model = model
        self.act, self.uses_grid = self._init_act(policy)

    def _init_act(self, policy):
        """
        Returns act function + whether or not the policy uses a grid layout
        (which discretizes the 2D space into a grid rather than operating
        in continuous space).
        """
        if policy == "straight_line":
            uses_grid = False
            act = self._straight_line_policy
        elif policy == "up_right":
            uses_grid = True
            act = self._up_right_policy
        elif policy == "right_up":
            uses_grid = True
            act = self._right_up_policy
        elif policy == "model":
            if self.model == None:
                return ValueError(f"For policy == model, self.model must be provided but got self.model == None!")
            uses_grid = False
            act = self._model_policy

        else:
            raise NotImplementedError(f"Policy {policy} not yet implemented for Reach2D environment!")

        return act, uses_grid

    def _model_policy(self, obs):
        return self.model.get_action(obs).detach()

    def _straight_line_policy(self, obs):
        curr_state = obs[:2]
        goal_state = obs[2:]
        act = goal_state - curr_state
        act = REACH2D_ACT_MAGNITUDE * act / torch.norm(act)

        return act

    def _up_right_policy(self, obs):
        curr_state = obs[:2]
        goal_state = obs[2:]
        curr_x, curr_y = curr_state
        goal_x, goal_y = goal_state

        if not torch.isclose(curr_y, goal_y):
            # First, align y-coordinate
            if curr_y < goal_y:
                act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
            elif curr_y > goal_y:
                act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
        elif not torch.isclose(curr_x, goal_x):
            # Then align x-coordinate (only if y-coordinate aligned already)
            if curr_x < goal_x:
                act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
            elif curr_x > goal_x:
                act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
        else:
            act = torch.zeros(REACH2D_ACT_DIM)

        return act

    def _right_up_policy(self, obs):
        curr_state = obs[:2]
        goal_state = obs[2:]
        curr_x, curr_y = curr_state
        goal_x, goal_y = goal_state

        if not torch.isclose(curr_x, goal_x):
            # First, align x-coordinate
            if curr_x < goal_x:
                act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
            elif curr_x > goal_x:
                act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
        elif not torch.isclose(curr_y, goal_y):
            # Then align y-coordinate (only if x-coordinate aligned already)
            if curr_y < goal_y:
                act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
            elif curr_y > goal_y:
                act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
        else:
            act = torch.zeros(REACH2D_ACT_DIM)

        return act
