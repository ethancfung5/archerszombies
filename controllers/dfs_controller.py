# controllers/dfs_controller.py

from .base_controller import BaseController

from kaz_helpers import (
    get_closest_zombie,
    get_priority_zombie,
    decide_knight_action,
    decide_knight_action_instant,
)


class DepthFirstStubController(BaseController):
    name = "DFS"

    def __init__(self):
        super().__init__()
        self.use_simple = True
        self.use_priority = True

    def __call__(self, obs, action_space, agent, t):
        if not agent.startswith("knight"):
            return action_space.sample()

        if self.use_priority:
            z = get_priority_zombie(obs)
        else:
            z = get_closest_zombie(obs)

        def log(msg: str):
            if t < 300 and agent.endswith("0"):
                print(f"[DFS] {agent} t={t}: {msg}")

        if self.use_simple:
            act, label = decide_knight_action_instant(obs, action_space, z, attack_dist=0.05, t=t)
        else:
            act, label = decide_knight_action(obs, action_space, z, attack_dist=0.05, t=t)

        if z is None:
            log(f"no zombies -> {label} (action {act})")
        else:
            log(
                f"zombie dist={z.dist:.3f}, dx={z.dx:.3f}, dy={z.dy:.3f} -> "
                f"{label} (action {act})"
            )

        return act