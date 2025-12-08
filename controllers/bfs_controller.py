# controllers/bfs_controller.py

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from .base_controller import BaseController
from .movement_helper import MovementHelper


Coord = Tuple[int, int]


class BreadthFirstStubController(BaseController):
    """
    Breadth-First Search (BFS) controller operating on a small agent-centric grid.

    For each decision step:
      • The agent identifies the closest visible zombie in its local observation.
      • The zombie’s relative (x, y) position is projected onto a discrete grid.
      • BFS is executed from the agent at (0, 0) to the projected grid cell.
      • Only the first step of the BFS path is converted back to a continuous waypoint.
      • MovementHelper is used to map this waypoint and the true zombie distance
        to a discrete action (movement + attack).

    Movement and attack thresholds are role-dependent and are encapsulated in
    MovementHelper.
    """

    name = "BFS (grid chase 2D)"

    CELL_SIZE = 0.1       # Grid cell size in environment units
    GRID_RADIUS = 10      # Grid covers coordinates in [-R, R] for both x and y
    DEBUG = False         # Enable to print detailed diagnostics

    # ----------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ----------------------------------------------------------------------
    def __call__(self, obs, action_space, agent, t):
        """
        Main entry point for the controller.

        Dispatches to a BFS-based policy depending on the agent role.
        """
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="archer")
        else:
            # Unknown agent type: default to a no-op style action.
            return MovementHelper.noop_or_default(action_space)

    # ----------------------------------------------------------------------
    # MAIN CONTROL LOGIC
    # ----------------------------------------------------------------------
    def _act_with_bfs(self, arr, action_space, agent, t, role: str):
        """
        Compute an action for the given role using BFS on the local grid.
        """
        label = f"BFS 2D {role.upper()}"

        if self.DEBUG:
            print("\n" + "=" * 70)
            print(f"[{label}] t = {t}, agent = {agent}")
            print(f"[{label}] obs shape = {arr.shape}")
            print("-" * 70)

        parsed = self._parse_obs_or_noop(arr)
        if parsed is None:
            if self.DEBUG:
                print(f"[{label}] invalid observation -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        typemasks, dists, rel, ang = parsed

        if self.DEBUG:
            self._debug_log_obs(arr, typemasks, role)

        # Agent state is stored in row 0
        self_heading_x, self_heading_y = ang[0]
        self_pos_x, self_pos_y = rel[0]

        if self.DEBUG:
            print(f"[{label}] self pos=({self_pos_x:.3f}, {self_pos_y:.3f})")
            print(f"[{label}] self heading=({self_heading_x:.3f}, {self_heading_y:.3f})")

        # ------------------------------------------------------------------
        # TARGET SELECTION
        # ------------------------------------------------------------------
        target_info = self._select_closest_zombie(typemasks, dists, rel, ang)
        if target_info is None:
            if self.DEBUG:
                print(f"[{label}] no zombies visible -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        closest_idx, dist_to_zombie, (z_rx, z_ry), _ = target_info

        if self.DEBUG:
            print(
                f"[{label}] closest zombie rel=({z_rx:.3f}, {z_ry:.3f}) "
                f"dist={dist_to_zombie:.3f}"
            )

        # ------------------------------------------------------------------
        # BFS PLANNING ON THE LOCAL GRID
        # ------------------------------------------------------------------
        next_step_rel = self._compute_bfs_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )

        # If BFS does not produce a valid path, fall back to direct pursuit.
        if next_step_rel is None:
            if self.DEBUG:
                print(f"[{label}] BFS did not produce a path -> direct chase")
            target_rel = (z_rx, z_ry)
        else:
            target_rel = next_step_rel
            if self.DEBUG:
                print(f"[{label}] next grid waypoint = {target_rel}")

        # ------------------------------------------------------------------
        # MOVEMENT AND ATTACK DECISION
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
    def _compute_bfs_step(
        self,
        goal_rel: Tuple[float, float],
        label: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Execute BFS on the local integer grid from (0, 0) to the projected goal cell.

        Returns the next waypoint in continuous local coordinates, or None if
        no path is available within the bounded grid.
        """
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        # Project goal into integer grid coordinates and clamp to grid bounds.
        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        if self.DEBUG:
            print(f"[{label}] BFS start={start}, goal={goal}")

        # If the projected goal coincides with the origin, use the original vector.
        if start == goal:
            return goal_rel

        path = self._bfs_grid_search(start, goal, R, label)
        if not path or len(path) < 2:
            return None

        # The first element is the start; the second is the next cell on the path.
        next_cell = path[1]
        nx, ny = next_cell

        # Convert back to continuous local coordinates.
        return (nx * cell_size, ny * cell_size)

    def _bfs_grid_search(
        self,
        start: Coord,
        goal: Coord,
        R: int,
        label: str,
    ) -> Optional[List[Coord]]:
        """
        Breadth-First Search on a bounded 4-connected grid.

        Grid states are integer pairs (x, y) with -R <= x, y <= R.
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
            print(f"[{label}] BFS did not reach goal={goal}")
        return None

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Coord, Optional[Coord]],
        current: Coord,
    ) -> List[Coord]:
        """
        Reconstruct a path from the start state to `current` using the parent map.
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----------------------------------------------------------------------
    # TARGET SELECTION
    # ----------------------------------------------------------------------
    def _select_closest_zombie(
        self,
        typemasks: np.ndarray,
        dists: np.ndarray,
        rel: np.ndarray,
        ang: np.ndarray,
    ):
        """
        Select the closest zombie by Euclidean distance.

        Assumptions:
          • Row 0 corresponds to the controlled agent.
          • Zombies are encoded in typemasks[:, 0] (value > 0.5).
          • Other entities (e.g., allies) use different type channels.
        """
        rows = typemasks.shape[0]

        zombie_mask = np.zeros(rows, dtype=bool)
        zombie_mask[1:] = typemasks[1:, 0] > 0.5  # ignore self at index 0
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
        Parse the raw observation matrix into (typemasks, dists, rel, ang).

        Expected layout per row:
          • columns 0..5   : typemasks (entity type indicators)
          • column  6      : scalar distance
          • columns 7..8   : rel_x, rel_y (relative position)
          • columns 9..10  : ang_x, ang_y (heading or relative angle)
        """
        if arr.ndim != 2 or arr.shape[1] < 11:
            return None
        return (
            arr[:, :6],       # typemasks
            arr[:, 6],        # distances
            arr[:, 7:9],      # relative positions (x, y)
            arr[:, 9:11],     # angular components (x, y)
        )

    def _debug_log_obs(self, arr: np.ndarray, typemasks: np.ndarray, role: str):
        """
        Print a truncated view of the observation for debugging purposes.
        """
        rows = arr.shape[0]
        maxp = min(rows, 5)
        print(f"[BFS 2D {role.upper()}] first {maxp} rows of observation:")
        for i in range(maxp):
            print(" ", arr[i])
