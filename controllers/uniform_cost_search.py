# controllers/uniform_cost_controller.py

from .base_controller import BaseController


class UniformCostStubController(BaseController):
    """
    Uniform-cost search placeholder.

    In a real UCS, you would:
      - maintain a priority queue (min-heap) of frontier states keyed by path cost
      - repeatedly expand the least-cost frontier node
      - generate successor states via a forward model of the environment
      - pick the first action along the best found path

    KAZ uses high-dimensional observations and we don't have a compact state
    or forward model here, so this stub just samples a random action.
    """
    name = "Uniform cost search (stub)"

    def __call__(self, obs, action_space, agent, t):
        # Placeholder: purely random behavior for now.
        return action_space.sample()
