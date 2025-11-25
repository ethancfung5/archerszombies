# controllers/dfs_controller.py

from .base_controller import BaseController
import random


class DepthFirstStubController(BaseController):
    # Dropdown display
    name = "DFS"

    def __init__(self, problem=None):
        super().__init__()
        # Optional search problem; can be set later from outside
        self.problem = problem

        # Planned sequence of actions from DFS
        self._plan = []
        # Track which agent the current plan belongs to
        self._current_agent = None

    # DFS implementation
    def _depth_first_search(self, start_state, is_goal_fn, successors_fn):
        # Implementation of DFS using a stack
        stack = [start_state]

        # Keep track of parent states
        parent = {start_state: None}

        # Keep track of visited states
        visited = set()

        while stack:
            state = stack.pop()

            if state in visited:
                continue
            visited.add(state)

            # Check if we found a goal
            if is_goal_fn(state):
                # If so, construct a plan back to the start
                actions = []
                cur = state
                while parent[cur] is not None:
                    prev_state, action = parent[cur]
                    actions.append(action)
                    cur = prev_state
                actions.reverse()
                return actions

            # Expand successors in depth-first order
            for next_state, action in successors_fn(state):
                if next_state not in parent:
                    parent[next_state] = (state, action)
                    stack.append(next_state)

        # If no goal is found, return an empty plan
        return []

    # DFS plan computation
    def _compute_plan_with_dfs(self, obs, agent):
        # If no problem is set, do nothing
        if self.problem is None:
            return []

        # If problem is set, run DFS
        start_state = self.problem.get_start_state(obs, agent)
        is_goal_fn = self.problem.is_goal
        successors_fn = self.problem.get_successors

        plan = self._depth_first_search(start_state, is_goal_fn, successors_fn)
        return plan

    # Main DFS controller
    def __call__(self, obs, action_space, agent, t):
        # If we have a new agent or no plan, compute one
        if agent != self._current_agent or not self._plan:
            self._plan = self._compute_plan_with_dfs(obs, agent)
            self._current_agent = agent

        # If we have a DFS plan, try to use it
        if self._plan:
            planned_action = self._plan.pop(0)

            # If we have an action mask, check that the planned action is legal
            if isinstance(obs, dict) and "action_mask" in obs:
                mask = obs["action_mask"]
                # Check that the planned action is legal
                if 0 <= planned_action < len(mask) and mask[planned_action]:
                    return planned_action
                else:
                    # If planned action is illegal, fall back to any legal action
                    legal_actions = [i for i, m in enumerate(mask) if m]
                    if legal_actions:
                        return random.choice(legal_actions)
                    # If there are no legal actions, sample randomly
                    return action_space.sample()

            # No mask: just return the planned action
            return planned_action

        # If we don't have a DFS plan, check if we have an action mask
        if isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"]
            legal_actions = [i for i, m in enumerate(mask) if m]
            if legal_actions:
                return random.choice(legal_actions)

        # If we don't have a DFS plan and no action mask, sample randomly
        return action_space.sample()
