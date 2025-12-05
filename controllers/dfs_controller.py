# controllers/dfs_controller.py

from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .base_controller import BaseController
from .movement_helper import MovementHelper


Coord = Tuple[int, int]


class DepthFirstStubController(BaseController):
    """
    Depth-First Search (DFS) pathfinding controller on a simple 2D grid
    in the agent's LOCAL frame.

    - Knights and archers both:
        * find the closest zombie,
        * run DFS on a local grid from agent (0,0) to the zombie's local position,
        * take the NEXT grid cell on that path as a waypoint,
        * steer toward that waypoint using MovementHelper.

    Observation format (vector_state + typemasks assumed):
      per-row: [typemask(6), dist, rel_x, rel_y, ang_x, ang_y]
      typemask: [zombie, archer, knight, sword, arrow, self]

    Action mapping:
        0 = forward
        1 = backward
        2 = counterclockwise
        3 = clockwise
        4 = attack
        5 = noop
    """

    name = "DFS (grid chase 2D)"

    CELL_SIZE = 0.1      # world units per grid cell in local frame
    GRID_RADIUS = 10     # plan on [-R..R] x [-R..R]

    # ------------------------- MAIN ENTRYPOINT -------------------------

    def __call__(self, obs, action_space, agent, t):
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_dfs(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_dfs(arr, action_space, agent, t, role="archer")
        else:
            # zombies/arrows/etc.
            return action_space.sample()

    # ------------------------- HIGH-LEVEL LOGIC ------------------------

    def _act_with_dfs(self, arr, action_space, agent, t, role: str):
        label = f"DFS 2D {role.upper()}"

        print("\n" + "=" * 70)
        print(f"[{label}] t = {t}, agent = {agent}")
        print(f"[{label}] raw obs shape = {arr.shape}, ndim={arr.ndim}")
        print("-" * 70)

        parsed = self._parse_vector_obs(arr, role)
        if parsed is None:
            return MovementHelper.noop_or_default(action_space)

        typemasks, dists, rel, ang, self_idx = parsed

        self._debug_log_obs(arr, typemasks, role)

        # ---- self row ----
        self_typemask = typemasks[self_idx]
        self_pos_x, self_pos_y = rel[self_idx]
        self_heading_x, self_heading_y = ang[self_idx]

        print(f"[{label}] self row = {self_idx}")
        print(f"[{label}] self typemask = {self_typemask}")
        print(f"[{label}] self pos      = ({self_pos_x:.3f},{self_pos_y:.3f})")
        print(f"[{label}] self heading  = ({self_heading_x:.3f},{self_heading_y:.3f})")

        # ---- choose target: closest zombie ----
        target_info = self._select_closest_zombie(typemasks, dists, rel, ang, role, self_idx)
        if target_info is None:
            print(f"[{label}] no zombies visible -> NOOP")
            return MovementHelper.noop_or_default(action_space)

        closest_idx, dist_to_zombie, (z_rx, z_ry), (zx_ang_x, zx_ang_y) = target_info

        print(
            f"[{label}] CLOSEST zombie row={closest_idx}, "
            f"dist={dist_to_zombie:.3f}, rel=({z_rx:.3f},{z_ry:.3f}), "
            f"ang=({zx_ang_x:.3f},{zx_ang_y:.3f})"
        )

        # ------------------------------------------------------------------
        # DFS planning in local frame
        # ------------------------------------------------------------------
        next_step_rel = self._compute_dfs_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )

        if next_step_rel is None:
            print(f"[{label}] DFS failed, falling back to direct chase.")
            target_rel = (z_rx, z_ry)
        else:
            step_rx, step_ry = next_step_rel
            target_rel = (step_rx, step_ry)
            step_dist = float(np.hypot(step_rx, step_ry))
            print(
                f"[{label}] next waypoint in local frame: "
                f"({step_rx:.3f},{step_ry:.3f}), step_dist={step_dist:.3f}"
            )

        # ------------------------------------------------------------------
        # Movement + attack
        # ------------------------------------------------------------------
        action = MovementHelper.steer_towards_target(
            role=role,
            dist=dist_to_zombie,
            target_rel=target_rel,
            self_heading=(self_heading_x, self_heading_y),
            action_space=action_space,
            label=label,
        )

        print("=" * 70)
        return action

    # ------------------------------------------------------------------
    #  DFS ON LOCAL GRID (goal-directed)
    # ------------------------------------------------------------------

    def _compute_dfs_step(
        self,
        goal_rel: Tuple[float, float],
        label: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Run DFS from (0,0) to goal_rel (rx, ry) in local coordinates.

        - Grid covers [-GRID_RADIUS .. +GRID_RADIUS].
        - Each cell = CELL_SIZE units.
        """
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        # clamp to window
        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        print(f"[{label}] planning (DFS) from {start} to {goal} on grid "
              f"[-{R}..{R}] with cell_size={cell_size}")

        if start == goal:
            print(f"[{label}] start == goal in grid; direct target.")
            return goal_rel

        path = self._dfs_grid_search(start, goal, R, label)

        if not path or len(path) < 2:
            print(f"[{label}] DFS found no path or trivial path={path}")
            return None

        next_cell = path[1]
        nx, ny = next_cell

        next_rx = nx * cell_size
        next_ry = ny * cell_size

        print(f"[{label}] DFS path (first few): {path[:5]}")
        return (next_rx, next_ry)

    def _dfs_grid_search(
        self,
        start: Coord,
        goal: Coord,
        R: int,
        label: str,
    ) -> Optional[List[Coord]]:
        """
        DFS on a bounded 4-neighbor grid [-R..R] x [-R..R], but we
        ORDER neighbors by distance to the goal so DFS dives toward
        the goal instead of straight down.
        """
        base_neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return -R <= x <= R and -R <= y <= R

        def heuristic(a: Coord, b: Coord) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx * dx + dy * dy) ** 0.5

        stack: List[Coord] = [start]
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        visited: Set[Coord] = {start}

        while stack:
            current = stack.pop()

            if current == goal:
                print(f"[{label}] DFS reached goal at {current}")
                return self._reconstruct_path(came_from, current)

            cx, cy = current

            # Build candidate neighbors
            candidates: List[Tuple[Coord, float]] = []
            for dx, dy in base_neighbors:
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny):
                    continue
                neighbor = (nx, ny)
                if neighbor in visited:
                    continue
                h = heuristic(neighbor, goal)
                candidates.append((neighbor, h))

            # Sort so that the *closest* neighbor ends up popped first.
            # We push in descending heuristic order so the smallest h is on top.
            candidates.sort(key=lambda x: x[1], reverse=True)

            for neighbor, _h in candidates:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

        print(f"[{label}] DFS exhausted frontier; no path to {goal}")
        return None

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Coord, Optional[Coord]],
        current: Coord,
    ) -> List[Coord]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    #  OBSERVATION UTILITIES
    # ------------------------------------------------------------------

    def _parse_vector_obs(self, arr, role):
        """
        Interpret obs as [N, 11]:

            [typemask(6), dist, rel_x, rel_y, ang_x, ang_y]

        Returns:
            (typemasks, dists, rel, ang, self_idx) or None.
        """
        if arr.ndim == 1:
            if arr.size % 11 == 0:
                arr = arr.reshape(-1, 11)
                print(f"[DFS 2D {role.upper()}] reshaped 1D obs -> {arr.shape}")
            else:
                print(f"[DFS 2D {role.upper()}] 1D obs of size {arr.size} cannot be reshaped to 11 cols")
                return None

        if arr.ndim != 2:
            print(f"[DFS 2D {role.upper()}] obs ndim != 2 after reshape, shape={arr.shape}")
            return None

        rows, cols = arr.shape
        if cols < 11:
            print(f"[DFS 2D {role.upper()}] cols={cols} < 11; cannot parse vector+typemask")
            return None

        typemasks = arr[:, :6]
        dists = arr[:, 6]
        rel = arr[:, 7:9]
        ang = arr[:, 9:11]

        self_bits = typemasks[:, 5] > 0.5
        self_indices = np.where(self_bits)[0]

        if self_indices.size == 0:
            self_idx = 0
            print(f"[DFS 2D {role.upper()}] no explicit self row; defaulting self_idx=0")
        else:
            self_idx = int(self_indices[0])
            print(f"[DFS 2D {role.upper()}] detected self_idx={self_idx} via typemask[5]")

        return typemasks, dists, rel, ang, self_idx

    def _debug_log_obs(self, arr, typemasks, role):
        rows = arr.shape[0]
        max_rows_to_print = min(rows, 5)
        print(f"[DFS 2D {role.upper()}] first {max_rows_to_print} rows of obs:")
        for i in range(max_rows_to_print):
            print(f"  row {i}: {arr[i]}")

        type_names = ["zombie", "archer", "knight", "sword", "arrow", "self"]
        counts = typemasks.sum(axis=0)
        print(f"[DFS 2D {role.upper()}] typemask counts (approx):")
        for name, cnt in zip(type_names, counts):
            print(f"  {name:7s}: {cnt:.1f}")

    def _select_closest_zombie(self, typemasks, dists, rel, ang, role, self_idx: int):
        rows = typemasks.shape[0]

        zombie_mask = typemasks[:, 0] > 0.5
        zombie_mask[self_idx] = False
        zombie_indices = np.where(zombie_mask)[0]

        print(f"[DFS 2D {role.upper()}] zombie rows = {list(zombie_indices)}")

        if zombie_indices.size == 0:
            return None

        for idx in zombie_indices:
            dist_i = dists[idx]
            rx_i, ry_i = rel[idx]
            ax_i, ay_i = ang[idx]
            print(
                f"    zombie row={idx}: dist={dist_i:.3f}, "
                f"rel=({rx_i:.3f},{ry_i:.3f}), ang=({ax_i:.3f},{ay_i:.3f})"
            )

        closest_idx = zombie_indices[np.argmin(dists[zombie_indices])]
        dist = dists[closest_idx]
        rx, ry = rel[closest_idx]
        zx_ang_x, zx_ang_y = ang[closest_idx]

        return closest_idx, dist, (rx, ry), (zx_ang_x, zx_ang_y)
