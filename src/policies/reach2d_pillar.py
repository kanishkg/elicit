import torch

from constants import REACH2D_ACT_DIM, REACH2D_ACT_MAGNITUDE


class Reach2DPillarPolicy:
    def __init__(self, policy, pillar) -> None:
        self.policy = policy
        self.pillar = pillar
        self.act = self._init_act(policy)

    def _init_act(self, policy):
        if policy == "over":
            return self._over_policy
        elif policy == "under":
            return self._under_policy
        else:
            raise NotImplementedError(f"Policy {policy} not yet implemented for Reach2DPillar environment!")

    def _over_policy(self, obs):
        curr_state = obs[:2]
        goal_state = obs[2:4]
        pillar_state = obs[4:]
        curr_x, curr_y = curr_state
        goal_x, goal_y = goal_state
        pillar_x, pillar_y = pillar_state  # top-left corner

        # If pillar overlaps x-coordinate
        if (curr_x >= pillar_x) and curr_x <= (pillar_x + self.pillar.width) and curr_y <= (pillar_y):
            # Move to left of pillar
            act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
        elif curr_y <= pillar_y and curr_x <= (pillar_x + self.pillar.width):
            # Move above pillar
            act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])

        # If goal is above pillar, then proceed as normal (align y then x)
        elif goal_y > pillar_y:
            # If goal is below pillar, align x then y
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

        else:
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

    def _under_policy(self, obs):
        curr_state = obs[:2]
        goal_state = obs[2:4]
        pillar_state = obs[4:]
        curr_x, curr_y = curr_state
        goal_x, goal_y = goal_state
        pillar_x, pillar_y = pillar_state  # top-left corner

        # If pillar overlaps y-coordinate
        if (
            (curr_y >= pillar_y - self.pillar.height)
            and curr_y <= (pillar_y)
            and curr_x <= (pillar_x + self.pillar.width)
        ):
            # Move to below pillar
            act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
        elif curr_x <= (pillar_x + self.pillar.width) and curr_y <= pillar_y:
            # Move to right of pillar
            act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])

        # If goal is to right of pillar, then proceed as normal (align x then y)
        elif goal_x > pillar_x + self.pillar.width:
            if not torch.isclose(curr_x, goal_x):
                # First align x-coordinate (only if y-coordinate aligned already)
                if curr_x < goal_x:
                    act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
                elif curr_x > goal_x:
                    act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
            elif not torch.isclose(curr_y, goal_y):
                # Then, align y-coordinate
                if curr_y < goal_y:
                    act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
                elif curr_y > goal_y:
                    act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
            else:
                act = torch.zeros(REACH2D_ACT_DIM)

        else:
            if not torch.isclose(curr_y, goal_y):
                # First align y-coordinate (only if x-coordinate aligned already)
                if curr_y < goal_y:
                    act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
                elif curr_y > goal_y:
                    act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
            elif not torch.isclose(curr_x, goal_x):
                # Then, align x-coordinate
                if curr_x < goal_x:
                    act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
                elif curr_x > goal_x:
                    act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
            else:
                act = torch.zeros(REACH2D_ACT_DIM)

        return act
