# controllers/dfs_controller.py

from .base_controller import BaseController


class DepthFirstStubController(BaseController):
    """
    DFS placeholder: without a forward model and compact state, we can’t
    do true DFS on KAZ. Behaves randomly for now.
    """
    name = "DFS (stub)"

    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()
