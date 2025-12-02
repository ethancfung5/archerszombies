# controllers/bfs_controller.py

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from .base_controller import BaseController
from .movement_helper import MovementHelper


Coord = Tuple[int, int]


class BreadthFirstStubController(BaseController):
    """
    TRUE BFS controller that plans on a small local grid toward the closest zombie.

    - Knights and archers both:
        * find the closest zombie,
        * run BFS on a local grid from agent (0,0) to the zombie's local rel_x, rel_y,
        * take the NEXT grid cell on that path as a waypoint,
        * steer toward that waypoint using MovementHelper.

    - Uses proper movement + attack geometry from MovementHelper.
    """

    name = "BFS (grid chase 2D)"

    CELL_SIZE = 0.1
    GRID_RADIUS = 10

    def __call__(self, obs, action_space, agent, t):
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_bfs(arr, action_space, agent, t, role="archer")
        else:
            return action_space.sample()

    # ----------------------------------------------------------------------
    # MAIN LOGIC
    # ----------------------------------------------------------------------
    def _act_with_bfs(self, arr, action_space, agent, t, role: str):
        label = f"BFS 2D {role.upper()}"

        print("\n" + "=" * 70)
        print(f"[{label}] t = {t}, agent = {agent}")
        print(f"[{label}] obs shape = {arr.shape}")
        print("-" * 70)

        parsed = self._parse_obs_or_noop(arr, action_space, role)
        if parsed is None:
            return MovementHelper.noop_or_default(action_space)

        typemasks, dists, rel, ang = parsed

        # Log context
        self._debug_log_obs(arr, typemasks, role)

        # Self state (row 0)
        self_heading_x, self_heading_y = ang[0]
        self_pos_x, self_pos_y = rel[0]

        print(f"[{label}] self pos=({self_pos_x:.3f},{self_pos_y:.3f})")
        print(f"[{label}] self heading=({self_heading_x:.3f},{self_heading_y:.3f})")

        # ------------------------------------------------------------------
        # FIND CLOSEST ZOMBIE
        # ------------------------------------------------------------------
        target_info = self._select_closest_zombie(typemasks, dists, rel, ang, role)
        if target_info is None:
            print(f"[{label}] no zombies visible -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        closest_idx, dist_to_zombie, (z_rx, z_ry), _ = target_info

        print(f"[{label}] closest zombie rel=({z_rx:.3f},{z_ry:.3f}) dist={dist_to_zombie:.3f}")

        # ------------------------------------------------------------------
        # BFS PLANNING
        # ------------------------------------------------------------------
        next_step_rel = self._compute_bfs_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )

        if next_step_rel is None:
            print(f"[{label}] BFS failed -> using direct chase")
            target_rel = (z_rx, z_ry)
        else:
            target_rel = next_step_rel
            print(f"[{label}] next waypoint = {target_rel}")

        # ------------------------------------------------------------------
        # MOVEMENT USING WAYPOINT (attack uses dist_to_zombie)
        # ------------------------------------------------------------------
        action = MovementHelper.steer_towards_target(
            role=role,
            dist=dist_to_zombie,         # attack distance based on REAL zombie
            target_rel=target_rel,       # but movement based on waypoint
            self_heading=(self_heading_x, self_heading_y),
            action_space=action_space,
            label=label,
        )

        print("=" * 70)
        return action

    # ----------------------------------------------------------------------
    # BFS PLANNING
    # ----------------------------------------------------------------------
    def _compute_bfs_step(self, goal_rel, label):
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        # Clamp goal to grid
        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        print(f"[{label}] BFS grid start={start} goal={goal}")

        if start == goal:
            return goal_rel

        path = self._bfs_grid_search(start, goal, R, label)
        if not path or len(path) < 2:
            return None

        next_cell = path[1]
        nx, ny = next_cell

        return (nx * cell_size, ny * cell_size)

    def _bfs_grid_search(self, start, goal, R, label):
        neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

        def in_bounds(x, y):
            return -R <= x <= R and -R <= y <= R

        queue = deque([start])
        came_from = {start: None}

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

        print(f"[{label}] BFS could not reach {goal}")
        return None

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----------------------------------------------------------------------
    # ZOMBIE SELECTION  (THIS WAS MISSING IN YOUR ERROR)
    # ----------------------------------------------------------------------
    def _select_closest_zombie(self, typemasks, dists, rel, ang, role):
        rows = typemasks.shape[0]

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
    # OBS PARSING
    # ----------------------------------------------------------------------
    def _parse_obs_or_noop(self, arr, action_space, role):
        if arr.ndim != 2 or arr.shape[1] < 11:
            return None
        return (
            arr[:, :6],        # typemasks
            arr[:, 6],         # dists
            arr[:, 7:9],       # rel_x, rel_y
            arr[:, 9:11],      # ang_x, ang_y
        )

    def _debug_log_obs(self, arr, typemasks, role):
        rows = arr.shape[0]
        maxp = min(rows, 5)
        print(f"[BFS 2D {role.upper()}] printing first {maxp} rows:")
        for i in range(maxp):
            print(" ", arr[i])
