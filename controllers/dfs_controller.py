# controllers/dfs_controller.py

from .base_controller import BaseController
import random

# Implement a internal search problem to drive DFS
class KazProblem:
    def __init__(self, N=10):
        # We’ll search from 0 up to N
        self.N = N

    def get_start_state(self, obs, agent):
        # For now, ignore obs/agent and always start at 0
        return 0

    def is_goal(self, state):
        # Goal is to reach state N
        return state == self.N

    def get_successors(self, state):
        # Simple chain: 0 -> 1 -> 2 -> ... -> N
        if state >= self.N:
            return []
        next_state = state + 1
        # Use next_state as the "action" (1, 2, 3, ...)
        return [(next_state, next_state)]



class DepthFirstStubController(BaseController):
    # This label appears in the dropdown
    name = "DFS"

    def __init__(self, problem=None):
        super().__init__()

        # If a problem is passed in (for testing), use it.
        # Otherwise, use our tiny KazProblem so DFS always has a graph.
        if problem is not None:
            self.problem = problem
        else:
            self.problem = KazProblem(N=10)

        # Each agent gets its own DFS plan: {agent_name: [actions...]}
        self._plans = {}

    def _depth_first_search(self, start_state, is_goal_fn, successors_fn):
        """
        Basic depth-first search using an explicit stack.
        Returns a list of actions from start_state to a goal, or [] if none.
        """
        stack = [start_state]
        parent = {start_state: None}  # state -> (prev_state, action)
        visited = set()

        while stack:
            state = stack.pop()

            if state in visited:
                continue
            visited.add(state)

            # Check goal
            if is_goal_fn(state):
                # Reconstruct actions by walking parent pointers backwards
                actions = []
                cur = state
                while parent[cur] is not None:
                    prev_state, action = parent[cur]
                    actions.append(action)
                    cur = prev_state
                actions.reverse()
                return actions

            # Expand successors
            for next_state, action in successors_fn(state):
                if next_state not in parent:
                    parent[next_state] = (state, action)
                    stack.append(next_state)

        # No goal found
        return []

    def _compute_plan_with_dfs(self, obs, agent):
        # Get start state and goal check functions
        start_state = self.problem.get_start_state(obs, agent)
        is_goal_fn = self.problem.is_goal
        successors_fn = self.problem.get_successors
        return self._depth_first_search(start_state, is_goal_fn, successors_fn)


    def __call__(self, obs, action_space, agent, t):
        # Small debug once, just to prove DFS is running
        if t == 0 and agent == "archer_0":
            print("[DFS] First call for agent:", agent, "| obs type:", type(obs))

        # Get this agent's current plan (if any)
        plan = self._plans.get(agent)

        # If this agent has no plan or the plan is empty, compute a new one
        if not plan:
            plan = self._compute_plan_with_dfs(obs, agent)
            self._plans[agent] = plan

        # If DFS produced a plan, use actions from it
        if plan:
            planned_action = plan.pop(0)  # consume first action

            # Make sure the planned action is a valid index for this action space
            n = getattr(action_space, "n", None)
            if n is not None and n > 0:
                planned_action = planned_action % n  # keep in [0, n-1]

            # Debug: show what DFS decided
            print(f"[DFS] Agent {agent} at t={t} -> action {planned_action}")

            return planned_action

        # Fallback: if DFS gave us no plan, just pick a random action
        return action_space.sample()

# Standalone test
if __name__ == "__main__":
    # Tiny example to test DFS without KAZ, using a slightly longer chain.
    class SimpleChainProblem:
        """
        Simple chain: states 0 -> 1 -> 2 -> ... -> N
        Goal is to reach N. Each action is just moving to the next index.
        """

        def __init__(self, N):
            self.N = N

        def get_start_state(self, obs, agent):
            return 0

        def is_goal(self, state):
            return state == self.N

        def get_successors(self, state):
            if state >= self.N:
                return []
            next_state = state + 1
            return [(next_state, next_state)]

    problem = SimpleChainProblem(N=5)
    dfs = DepthFirstStubController(problem=problem)

    # Fake obs/agent
    obs = {}
    agent = "agent_0"
    plan = dfs._compute_plan_with_dfs(obs, agent)

    print("DFS plan:", plan)
    # Expected something like: [1, 2, 3, 4, 5]
