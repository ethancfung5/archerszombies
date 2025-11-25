# controllers/astar_controller.py

import heapq
from typing import Dict, List, Tuple, Optional

import numpy as np

from .base_controller import BaseController


Coord = Tuple[int, int]


class AStarStubController(BaseController):
    """
    A* pathfinding controller that chases zombies using a simple internal grid
    and a heading-based motion model that matches KAZ's action mapping:

        0: move forward (up)
        1: move backward (down)
        2: rotate counter-clockwise
        3: rotate clockwise
        4: attack
        5: stay still / idle

    Assumptions about the observation (vector_state=True, use_typemasks=False):

        - obs is an (N+1, 5) array.
        - Row 0 is the current agent.
        - The last `max_zombies` rows correspond to zombie slots.
        - For each zombie row:
            [distance, rel_dx, rel_dy, ..., ...]
          where rel_dx, rel_dy are normalized relative positions.

    High-level behavior per agent:

        1. Parse obs to find the nearest zombie from the tail rows.
        2. Map its relative position (dx, dy) into a goal cell on an internal grid.
        3. Use A* on that grid to plan a shortest path (in grid steps).
        4. At each step:
             - If not facing the next grid step, rotate one step toward it.
             - If already facing it, move forward.
        5. If close enough to the zombie, issue an attack.
        6. If no zombies visible, stay still (true idle).
    """

    name = "A* search"

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        max_zombies: int = 10,  # default KAZ value
    ):
        # Internal grid size (purely for planning)
        self.width = width
        self.height = height
        self.max_zombies = max_zombies

        # Per-agent state
        self.agent_pos: Dict[str, Coord] = {}       # internal grid position
        self.agent_goal: Dict[str, Coord] = {}      # last goal cell
        self.agent_paths: Dict[str, List[Coord]] = {}  # planned path (list of cells)
        self.agent_heading: Dict[str, int] = {}     # 0: up, 1: right, 2: down, 3: left

        # Movement directions in grid coordinates
        # These are *abstract* directions; actual game uses heading + forward movement.
        self.dir_vectors: Dict[int, Coord] = {
            0: (0, -1),   # up
            1: (1, 0),    # right
            2: (0, 1),    # down
            3: (-1, 0),   # left
        }

    # ------------------------------------------------------------------
    # A* core on the internal grid
    # ------------------------------------------------------------------

    def _in_bounds(self, p: Coord) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def _neighbors(self, p: Coord) -> List[Coord]:
        x, y = p
        result = []
        for dx, dy in self.dir_vectors.values():
            nx, ny = x + dx, y + dy
            np_ = (nx, ny)
            if self._in_bounds(np_):
                result.append(np_)
        return result

    def _heuristic(self, a: Coord, b: Coord) -> float:
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self, came_from: Dict[Coord, Coord], start: Coord, goal: Coord
    ) -> List[Coord]:
        cur = goal
        path_reversed = [cur]
        while cur != start:
            cur = came_from[cur]
            path_reversed.append(cur)
        path_reversed.reverse()
        return path_reversed

    def _a_star(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        open_heap: List[Tuple[float, Coord]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[Coord, Coord] = {}
        g_score: Dict[Coord, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, start, goal)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + 1.0
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, neighbor))

        return None

    # ------------------------------------------------------------------
    # KAZ vector_state parsing (no typemasks)
    # ------------------------------------------------------------------

    def _find_nearest_zombie(self, obs: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        For default KAZ vector_state (no typemasks):

          - obs is an (N+1, 5) array.
          - Row 0 is current agent.
          - The last `max_zombies` rows correspond to zombie slots.

        We:
          - look at the last max_zombies rows,
          - ignore rows that are all zeros (no entity),
          - treat them as zombies,
          - return (distance, dx, dy) of the closest one.
        """

        #print(obs)
        if obs.ndim != 2:
            return None

        num_rows, width = obs.shape
        if width != 5:
            # Not the vector_state layout we expect
            return None

        if num_rows <= 1:
            return None

        # Ignore row 0 (current agent)
        body = obs[1:]
        print(body, "\n")

        # Take the last `max_zombies` rows as zombie slots.
        start_idx = max(0, body.shape[0] - self.max_zombies)
        zombie_block = body[start_idx:]

        # Drop rows that are all zeros (no entity)
        mask_nonzero = np.any(zombie_block != 0.0, axis=1)
        zombie_rows = zombie_block[mask_nonzero]

        if zombie_rows.size == 0:
            return None

        distances = zombie_rows[:, 0]
        rel = zombie_rows[:, 1:3]  # (dx, dy)
        nonzero = distances > 0.0
        if not np.any(nonzero):
            return None

        distances = distances[nonzero]
        rel = rel[nonzero]
        idx = int(np.argmin(distances))
        d = float(distances[idx])
        dx, dy = rel[idx]
        return d, float(dx), float(dy)

    def _relative_pos_to_grid_goal(self, dx: float, dy: float) -> Coord:
        """
        Map continuous relative position (dx, dy) into an internal grid goal
        with the agent at the center.
        """
        cx = self.width // 2
        cy = self.height // 2

        # Heuristic scaling: KAZ relative positions are normalized.
        SCALE = 6.0
        gx = cx + int(round(dx * SCALE))
        gy = cy + int(round(dy * SCALE))

        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy

    # ------------------------------------------------------------------
    # Per-agent state helpers
    # ------------------------------------------------------------------

    def _ensure_agent_state(self, agent: str) -> Tuple[Coord, int]:
        """
        Ensure we have an internal position and heading for this agent.

        Heading convention:
            0: up
            1: right
            2: down
            3: left
        """
        if agent not in self.agent_pos:
            self.agent_pos[agent] = (self.width // 2, self.height // 2)
        if agent not in self.agent_heading:
            self.agent_heading[agent] = 0  # facing up by default
        if agent not in self.agent_paths:
            self.agent_paths[agent] = []
        return self.agent_pos[agent], self.agent_heading[agent]

    def _plan_to_goal(self, agent: str, start: Coord, goal: Coord):
        """
        Plan A* path for this agent from start to goal.
        """
        self.agent_goal[agent] = goal
        path = self._a_star(start, goal)
        if path is None or len(path) <= 1:
            self.agent_paths[agent] = []
        else:
            self.agent_paths[agent] = path

    # ------------------------------------------------------------------
    # Heading-based control: path -> discrete actions
    # ------------------------------------------------------------------

    def _rotate_towards_dir(self, agent: str, desired_dir: int, action_space) -> int:
        """
        Rotate the agent one step (CCW or CW) toward the desired_dir.

        Returns an action (2: CCW, 3: CW) and updates internal heading.
        """
        if not hasattr(action_space, "n"):
            return action_space.sample()

        heading = self.agent_heading.get(agent, 0)
        # delta in [0..3], desired_dir = heading + delta (mod 4)
        delta = (desired_dir - heading) % 4

        if delta == 0:
            # Already facing the desired direction; nothing to rotate.
            return 5 if action_space.n > 5 else 0

        # Choose shortest rotation direction.
        # delta = 1 => 1 step CW
        # delta = 3 => 1 step CCW
        # delta = 2 => either; pick CW for consistency (2 rotations total over time).
        if delta == 1 or delta == 2:
            # Rotate CW
            action = 3  # rotate clockwise
            self.agent_heading[agent] = (heading + 1) % 4
        elif delta == 3:
            # Rotate CCW
            action = 2  # rotate counter-clockwise
            self.agent_heading[agent] = (heading - 1) % 4
        else:
            # Fallback, should not happen
            action = 5 if action_space.n > 5 else 0

        return action

    def _forward_action(self, action_space) -> int:
        """
        Move forward (up) in the agent's current heading frame.

        In KAZ mapping:
            0: move forward (up)
        """
        if not hasattr(action_space, "n"):
            return action_space.sample()
        if action_space.n > 0:
            return 0  # move forward
        return action_space.sample()

    def _attack_action(self, action_space) -> int:
        """
        Attack action according to the KAZ mapping.
        """
        if not hasattr(action_space, "n"):
            return action_space.sample()
        if action_space.n > 4:
            return 4
        return 0

    def _idle_action(self, action_space) -> int:
        """
        Stay still / idle action according to the KAZ mapping.
        """
        if not hasattr(action_space, "n"):
            return action_space.sample()
        if action_space.n > 5:
            return 5  # stay still
        # If no explicit idle, default to "do nothing-ish"
        return 0

    def _next_action_from_path(self, agent: str, action_space) -> int:
        """
        Follow the planned path while respecting heading:

          - Let path[0] be the current cell, path[1] the next.
          - Compute which abstract direction (up/right/down/left) that step corresponds to.
          - If the agent is not facing that direction:
                rotate one step (2 or 3) and keep the path as-is.
          - If the agent is facing it:
                move forward (0), update internal position and pop the first cell.
        """
        if action_space is None or not hasattr(action_space, "n"):
            return action_space.sample()

        path = self.agent_paths.get(agent, [])
        if len(path) < 2:
            # No path or already at goal
            return self._idle_action(action_space)

        # Current and next grid cells
        cur = path[0]
        nxt = path[1]
        dx = nxt[0] - cur[0]
        dy = nxt[1] - cur[1]

        # Determine desired direction index based on (dx, dy)
        desired_dir: Optional[int] = None
        for dir_idx, vec in self.dir_vectors.items():
            if vec == (dx, dy):
                desired_dir = dir_idx
                break

        if desired_dir is None:
            # Non-cardinal or weird step; just idle this turn
            return self._idle_action(action_space)

        heading = self.agent_heading.get(agent, 0)

        if heading != desired_dir:
            # Rotate one step towards the desired heading
            return self._rotate_towards_dir(agent, desired_dir, action_space)

        # Already facing the correct direction: move forward
        action = self._forward_action(action_space)

        # Update internal position and remaining path
        self.agent_pos[agent] = nxt
        # Drop the first cell so path[0] remains "current"
        self.agent_paths[agent] = path[1:]

        return action

    # ------------------------------------------------------------------
    # Controller entry point
    # ------------------------------------------------------------------

    def __call__(self, obs, action_space, agent, t):
        """
        Main controller logic:

          - If no zombie visible → stay still (true idle).
          - Otherwise:
              * find nearest zombie (distance, dx, dy)
              * map to grid goal
              * plan (or replan) with A*
              * if close enough, attack
              * else follow the path using heading-aware rotation + forward
        """
        # Safely convert obs
        try:
            obs_arr = np.array(obs)
        except Exception as e:
            print(f"[A*] Agent {agent}: could not convert obs to array: {e}")
            return self._idle_action(action_space)

        # 1. Find nearest zombie from tail rows of the vector state
        nearest = self._find_nearest_zombie(obs_arr)
        if nearest is None:
            # No zombies visible: stay still so agents don't sprint randomly
            return self._idle_action(action_space)

        distance, dx, dy = nearest

        # 2. Build / update internal planning state
        start, _ = self._ensure_agent_state(agent)
        goal = self._relative_pos_to_grid_goal(dx, dy)

        prev_goal = self.agent_goal.get(agent)
        path = self.agent_paths.get(agent, [])

        # Replan if goal changed or we have no path
        if (not path) or (prev_goal != goal):
            self._plan_to_goal(agent, start, goal)
            path = self.agent_paths.get(agent, [])

        # 3. If we're close to the goal, try to attack instead of moving
        grid_dist = self._heuristic(start, goal)
        if grid_dist <= 1 or distance <= 0.08:
            return self._attack_action(action_space)

        # 4. Otherwise follow the path with heading-aware control
        return self._next_action_from_path(agent, action_space)
