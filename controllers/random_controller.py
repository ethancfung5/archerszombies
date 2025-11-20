# controllers/random_controller.py

from .base_controller import BaseController


class RandomController(BaseController):
    name = "Random"

    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()
