# controllers/dfs_controller.py

from .base_controller import BaseController
import random


class DepthFirstStubController(BaseController):
    # Dropdown display
    name = "DFS"

    def __init__(self, problem=None):
        super().__init__()
        # Optional search problem; can be set later from outside
        if problem is not None:
            self.problem = problem
        else:
            self.problem = KazProblem()
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
        if t == 0 and agent == "archer_0":
            print("[DFS] First call for agent:", agent)
            print("    type(obs):", type(obs))
            if isinstance(obs, dict):
                print("    obs keys:", list(obs.keys()))
        if t == 0 and agent == "knight_0":  # or any agent name you see
            print("OBS TYPE:", type(obs))
            if isinstance(obs, dict):
                print("OBS KEYS:", list(obs.keys()))
            print("FULL OBS:", obs)
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

if __name__ == "__main__":
    # Tiny example to test DFS without KAZ

    class SimpleChainProblem:
        """
        Simple chain: states 0 -> 1 -> 2 -> ... -> N
        Goal is to reach N. Each action is just moving to the next index.
        """

        def __init__(self, N):
            self.N = N

        def get_start_state(self, obs, agent):
            # For this toy example, ignore obs and agent
            return 0

        def is_goal(self, state):
            return state == self.N

        def get_successors(self, state):
            if state >= self.N:
                return []
            # Next state is state+1; action is also (state+1) just for demo
            return [(state + 1, state + 1)]

    # Import controller here
    #from dfs_controller import DepthFirstStubController  # adjust import if needed

    problem = SimpleChainProblem(N=5)
    dfs = DepthFirstStubController(problem=problem)

    # Fake obs/agent
    obs = {}
    agent = "agent_0"
    plan = dfs._compute_plan_with_dfs(obs, agent)

    print("DFS plan:", plan)
    # Expected something like: [1, 2, 3, 4, 5]

class KazProblem:
    def __init__(self, N=3):
        self.N = N

    def get_start_state(self, obs, agent):
        # For now, we ignore obs and agent and always start at 0.
        return 0

    def is_goal(self, state):
        # Goal is to reach state N.
        return state == self.N

    def get_successors(self, state):
        if state >= self.N:
            return []
        next_state = state + 1
        # Action is just the next state's index for this toy example
        return [(next_state, next_state)]

