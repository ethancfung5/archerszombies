# controllers/astar_controller.py

from .base_controller import BaseController


class AStarStubController(BaseController):
    """
    A* placeholder: for real A* you’d need an internal map/state extractor
    or a model-based simulator.
    """
    name = "A* (stub)"

    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()
