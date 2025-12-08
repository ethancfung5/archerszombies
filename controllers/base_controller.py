# controllers/base_controller.py

class BaseController:
    name = "Base"

    def __call__(self, obs, action_space, agent, t):
        try:
            return action_space.sample()
        except Exception:
            # Very last resort if something weird happens
            return 0
