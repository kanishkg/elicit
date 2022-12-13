from policies.base_robosuite_policy import BaseRobosuitePolicy


class NutAssemblyPolicy(BaseRobosuitePolicy):
    def __init__(self, policy, env, robosuite_cfg) -> None:
        super().__init__(policy, env, robosuite_cfg)

        self.act = self._init_act()

    def _init_act(self):
        if self.policy == "user":
            return self._user_policy
        else:
            raise NotImplementedError(f"Policy {self.policy} not yet implemented for Reach2DPillar environment!")
