# controllers/base_controller.py

class BaseController:
    name = "Base"

    def __call__(self, obs, action_space, agent, t):
        raise NotImplementedError
