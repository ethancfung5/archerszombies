# controllers/bfs_controller.py

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from .base_controller import BaseController
from .movement_helper import MovementHelper


Coord = Tuple[int, int]


class BreadthFirstStubController(BaseController):
    """
    Breadth-First Search (BFS) controller that plans on a small agent-centric grid
    toward the closest visible zombie.

    High-level behaviour
    --------------------
    - For both knights and archers:
        * Identify the closest zombie in the local observation.
        * Project the zombie's relative (x, y) position onto a discrete grid.
        * Run BFS from the agent at (0, 0) to that grid cell.
        * Take ONLY the next grid cell as a waypoint in local coordinates.
        * Use MovementHelper to steer toward the waypoint and decide whether to attack.

    - Movement and attack thresholds are role-dependent and implemented inside
      MovementHelper.
    """

    name = "BFS (grid chase 2D)"

    CELL_SIZE = 0.1       # size of each grid cell in environment units
    GRID_RADIUS = 10      # grid spans [-R, R] in both x and y
    DEBUG = False         # set to True for detailed console logging

    # ----------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ----------------------------------------------------------------------
    def __call__(self, obs, action_space, agent, t):
        """
        Main controller entry. Dispatches to a role-specific BFS policy.
        """
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="archer")
        else:
            # Fallback for unexpected agent types.
            return MovementHelper.noop_or_default(action_space)

    # ----------------------------------------------------------------------
    # MAIN LOGIC
    # ----------------------------------------------------------------------
    def _act_with_bfs(self, arr, action_space, agent, t, role: str):
        label = f"BFS 2D {role.upper()}"

        if self.DEBUG:
            print("\n" + "=" * 70)
            print(f"[{label}] t = {t}, agent = {agent}")
            print(f"[{label}] obs shape = {arr.shape}")
            print("-" * 70)

        parsed = self._parse_obs_or_noop(arr)
        if parsed is None:
            if self.DEBUG:
                print(f"[{label}] invalid obs -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        typemasks, dists, rel, ang = parsed

        if self.DEBUG:
            self._debug_log_obs(arr, typemasks, role)

        # Self state (row 0)
        self_heading_x, self_heading_y = ang[0]
        self_pos_x, self_pos_y = rel[0]

        if self.DEBUG:
            print(f"[{label}] self pos=({self_pos_x:.3f},{self_pos_y:.3f})")
            print(f"[{label}] self heading=({self_heading_x:.3f},{self_heading_y:.3f})")

        # ------------------------------------------------------------------
        # SELECT TARGET ZOMBIE
        # ------------------------------------------------------------------
        target_info = self._select_closest_zombie(typemasks, dists, rel, ang)
        if target_info is None:
            if self.DEBUG:
                print(f"[{label}] no zombies visible -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        closest_idx, dist_to_zombie, (z_rx, z_ry), _ = target_info

        if self.DEBUG:
            print(
                f"[{label}] closest zombie rel=({z_rx:.3f},{z_ry:.3f}) "
                f"dist={dist_to_zombie:.3f}"
            )

        # ------------------------------------------------------------------
        # BFS PLANNING ON LOCAL GRID
        # ------------------------------------------------------------------
        next_step_rel = self._compute_bfs_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )

        # If BFS fails (e.g., numerical edge cases), fall back to direct chase.
        if next_step_rel is None:
            if self.DEBUG:
                print(f"[{label}] BFS failed -> using direct chase")
            target_rel = (z_rx, z_ry)
        else:
            target_rel = next_step_rel
            if self.DEBUG:
                print(f"[{label}] next waypoint = {target_rel}")

        # ------------------------------------------------------------------
        # MOVEMENT + ATTACK
        #   - Movement uses the BFS waypoint.
        #   - Attack uses the *true* distance to the zombie.
        # ------------------------------------------------------------------
        action = MovementHelper.steer_towards_target(
            role=role,
            dist=dist_to_zombie,
            target_rel=target_rel,
            self_heading=(self_heading_x, self_heading_y),
            action_space=action_space,
            label=label if self.DEBUG else None,
        )

        if self.DEBUG:
            print("=" * 70)

        return action

    # ----------------------------------------------------------------------
    # BFS PLANNING
    # ----------------------------------------------------------------------
    def _compute_bfs_step(self, goal_rel: Tuple[float, float], label: str
                          ) -> Optional[Tuple[float, float]]:
        """
        Run BFS on a small integer grid from (0, 0) to the projected goal cell.
        Returns the *next* waypoint in continuous local coordinates,
        or None if no path is found.
        """
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        # Project goal onto grid and clamp to local planning radius.
        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        if self.DEBUG:
            print(f"[{label}] BFS grid start={start} goal={goal}")

        # If the projected goal is at the origin, just move directly toward it.
        if start == goal:
            return goal_rel

        path = self._bfs_grid_search(start, goal, R, label)
        if not path or len(path) < 2:
            return None

        next_cell = path[1]
        nx, ny = next_cell

        # Convert back into local continuous coordinates.
        return (nx * cell_size, ny * cell_size)

    def _bfs_grid_search(self, start: Coord, goal: Coord, R: int, label: str
                         ) -> Optional[List[Coord]]:
        """
        Standard BFS on a bounded 4-connected grid:
        states are integer (x, y) with -R <= x,y <= R.
        """
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return -R <= x <= R and -R <= y <= R

        queue = deque([start])
        came_from: Dict[Coord, Optional[Coord]] = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self._reconstruct_path(came_from, current)

            cx, cy = current
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny):
                    continue
                neighbor = (nx, ny)
                if neighbor in came_from:
                    continue
                came_from[neighbor] = current
                queue.append(neighbor)

        if self.DEBUG:
            print(f"[{label}] BFS could not reach {goal}")
        return None

    @staticmethod
    def _reconstruct_path(came_from: Dict[Coord, Optional[Coord]],
                          current: Coord) -> List[Coord]:
        """
        Reconstruct path from start to current using the 'came_from' map.
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----------------------------------------------------------------------
    # ZOMBIE SELECTION
    # ----------------------------------------------------------------------
    def _select_closest_zombie(self,
                               typemasks: np.ndarray,
                               dists: np.ndarray,
                               rel: np.ndarray,
                               ang: np.ndarray):
        """
        Select the closest zombie based on distance.

        Assumes:
        - Row 0 is the agent itself.
        - Zombies are encoded in typemasks[:, 0] (first type channel > 0.5).
        - Other agents (knights/archers) occupy different type channels and
          are therefore ignored.
        """
        rows = typemasks.shape[0]

        # Mask out self (row 0) and select rows where "zombie" channel is active.
        zombie_mask = np.zeros(rows, dtype=bool)
        zombie_mask[1:] = typemasks[1:, 0] > 0.5
        zombie_indices = np.where(zombie_mask)[0]

        if zombie_indices.size == 0:
            return None

        closest_idx = zombie_indices[np.argmin(dists[zombie_indices])]
        dist = dists[closest_idx]
        rx, ry = rel[closest_idx]
        ax, ay = ang[closest_idx]

        return closest_idx, dist, (rx, ry), (ax, ay)

    # ----------------------------------------------------------------------
    # OBSERVATION PARSING
    # ----------------------------------------------------------------------
    def _parse_obs_or_noop(self, arr: np.ndarray):
        """
        Parse the raw observation into (typemasks, dists, rel, ang).

        Expected layout per row:
        - columns 0..5   : typemasks (entity type one-hot / soft mask)
        - column  6      : distance
        - columns 7..8   : rel_x, rel_y
        - columns 9..10  : ang_x, ang_y
        """
        if arr.ndim != 2 or arr.shape[1] < 11:
            return None
        return (
            arr[:, :6],       # typemasks
            arr[:, 6],        # dists
            arr[:, 7:9],      # rel_x, rel_y
            arr[:, 9:11],     # ang_x, ang_y
        )

    def _debug_log_obs(self, arr: np.ndarray, typemasks: np.ndarray, role: str):
        """
        Debug helper to print the first few rows of the observation.
        """
        rows = arr.shape[0]
        maxp = min(rows, 5)
        print(f"[BFS 2D {role.upper()}] printing first {maxp} rows:")
        for i in range(maxp):
            print(" ", arr[i])
