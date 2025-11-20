# controllers/bfs_controller.py

from .base_controller import BaseController


class BreadthFirstStubController(BaseController):
    name = "BFS (stub)"

    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()
