# controllers/astar_controller.py

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_controller import BaseController
from .movement_helper import MovementHelper


Coord = Tuple[int, int]


class AStarController(BaseController):
    """
    A* pathfinding controller on a simple 2D grid in the agent's LOCAL frame.

    - Knights and archers both:
        * find the closest zombie,
        * run A* on a local grid from agent (0,0) to the zombie's local position,
        * take the NEXT grid cell on that path as a waypoint,
        * steer toward that waypoint using 2D geometry via MovementHelper.

    - Same attack logic as BFS (via MovementHelper):
        * Knights attack at a small distance (melee).
        * Archers attack at a larger distance (ranged).

    Observation format (vector_state + typemasks assumed):
      row: [typemask(6), dist, rel_x, rel_y, ang_x, ang_y]
      typemask: [zombie, archer, knight, sword, arrow, self]

    Action mapping:
        0 = forward
        1 = backward
        2 = counterclockwise
        3 = clockwise
        4 = attack
        5 = noop
    """

    name = "A* (grid chase 2D)"

    # Grid planning parameters
    CELL_SIZE = 0.1      # world units per grid cell in local frame
    GRID_RADIUS = 10     # plan on [-R..R] x [-R..R] --> (2R+1)^2 grid

    # ------------------------- MAIN ENTRYPOINT -------------------------

    def __call__(self, obs, action_space, agent, t):
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_astar(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_astar(arr, action_space, agent, t, role="archer")
        else:
            # Anything else (zombies, arrows, etc.) – just random
            return action_space.sample()

    # ------------------------- HIGH-LEVEL LOGIC ------------------------

    def _act_with_astar(self, arr, action_space, agent, t, role: str):
        """
        Shared chasing logic for knights and archers using A*.
        """
        label = f"A* 2D {role.upper()}"

        print("\n" + "=" * 70)
        print(f"[{label}] t = {t}, agent = {agent}")
        print(f"[{label}] obs shape = {arr.shape}")
        print("-" * 70)

        # ---- Sanity checks + basic slicing ----
        parsed = self._parse_obs_or_noop(arr, action_space, role)
        if parsed is None:
            # Already logged; NOOP
            return MovementHelper.noop_or_default(action_space)

        typemasks, dists, rel, ang = parsed

        # Optional: log first few rows / type counts
        self._debug_log_obs(arr, typemasks, role)

        # ---- Self row (row 0) ----
        self_typemask = typemasks[0]
        self_pos_x, self_pos_y = rel[0]   # usually (0,0) in local frame
        self_heading_x, self_heading_y = ang[0]

        print(f"[{label}] row 0 (self) breakdown:")
        print(f"  typemask = {self_typemask}")
        print(f"  pos      = ({self_pos_x:.3f},{self_pos_y:.3f})")
        print(f"  heading  = ({self_heading_x:.3f},{self_heading_y:.3f})")

        # ---- Choose target: closest zombie ----
        target_info = self._select_closest_zombie(typemasks, dists, rel, ang, role)
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
        # A* PLANNING in LOCAL frame:
        #   - start = (0,0)  (agent)
        #   - goal  = (z_rx, z_ry) (zombie position in local coords)
        # ------------------------------------------------------------------
        next_step_rel = self._compute_astar_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )

        # Fallback: if A* failed, just chase directly
        if next_step_rel is None:
            print(f"[{label}] A* failed, falling back to direct chase.")
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
        # MOVEMENT + ATTACK: reuse shared steering logic
        #
        # IMPORTANT: use dist_to_zombie for attack range, not distance to waypoint.
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
    #  A* ON LOCAL GRID
    # ------------------------------------------------------------------

    def _compute_astar_step(
        self,
        goal_rel: Tuple[float, float],
        label: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Run A* from agent (0,0) to goal_rel (rx, ry) in local coordinates.

        - Grid covers [-GRID_RADIUS .. +GRID_RADIUS] in each axis.
        - Each cell = CELL_SIZE in world units.
        - Currently: all cells are traversable (no obstacles yet).

        Returns:
            next_step_rel = (rx, ry) for the NEXT waypoint in local coords,
            or None if planning fails.
        """
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        # Clamp to planning window
        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        print(f"[{label}] planning (A*) from {start} to {goal} on grid "
              f"[-{R}..{R}] with cell_size={cell_size}")

        if start == goal:
            print(f"[{label}] start == goal in grid; direct target.")
            return goal_rel

        path = self._astar_grid_search(start, goal, R, label)

        if not path or len(path) < 2:
            print(f"[{label}] A* found no path or trivial path={path}")
            return None

        # path[0] is start, path[1] is next cell on path
        next_cell = path[1]
        nx, ny = next_cell

        # Convert grid cell back to local coordinates
        next_rx = nx * cell_size
        next_ry = ny * cell_size

        return (next_rx, next_ry)

    def _astar_grid_search(
        self,
        start: Coord,
        goal: Coord,
        R: int,
        label: str,
    ) -> Optional[List[Coord]]:
        """
        Classic A* on a bounded 4-neighbor grid [-R..R] x [-R..R].

        All cells are currently traversable (no obstacles).
        """
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return -R <= x <= R and -R <= y <= R

        def heuristic(a: Coord, b: Coord) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx * dx + dy * dy) ** 0.5

        open_heap: List[Tuple[float, Coord]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        g_score: Dict[Coord, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                print(f"[{label}] A* reached goal at {current}")
                return self._reconstruct_path(came_from, current)

            cx, cy = current

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny):
                    continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1.0  # uniform cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        # No path
        print(f"[{label}] A* exhausted frontier; no path to {goal}")
        return None

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Coord, Optional[Coord]],
        current: Coord,
    ) -> List[Coord]:
        """
        Reconstruct path (from start to current) using came_from map.
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    #  OBSERVATION UTILITIES (same idea as BFS)
    # ------------------------------------------------------------------

    def _parse_obs_or_noop(self, arr, action_space, role):
        """
        Validate observation shape and return (typemasks, dists, rel, ang),
        or None if invalid.
        """
        if arr.ndim != 2:
            print(f"[A* 2D {role.upper()}] obs ndim != 2, shape={arr.shape}; returning NOOP")
            return None

        rows, cols = arr.shape
        if cols < 11:
            print(f"[A* 2D {role.upper()}] cols={cols} < 11; not vector+typemask; returning NOOP")
            return None

        typemasks = arr[:, :6]
        dists = arr[:, 6]
        rel = arr[:, 7:9]   # (rel_x, rel_y)
        ang = arr[:, 9:11]  # (ang_x, ang_y)

        return typemasks, dists, rel, ang

    def _debug_log_obs(self, arr, typemasks, role):
        """
        Log a few rows and type counts for debugging.
        """
        rows = arr.shape[0]
        max_rows_to_print = min(rows, 5)
        print(f"[A* 2D {role.upper()}] first {max_rows_to_print} rows of obs:")
        for i in range(max_rows_to_print):
            print(f"  row {i}: {arr[i]}")

        type_names = ["zombie", "archer", "knight", "sword", "arrow", "self"]
        counts = typemasks.sum(axis=0)
        print(f"[A* 2D {role.upper()}] typemask counts (approx):")
        for name, cnt in zip(type_names, counts):
            print(f"  {name:7s}: {cnt:.1f}")

    def _select_closest_zombie(self, typemasks, dists, rel, ang, role):
        """
        Return (closest_idx, dist, (rx, ry), (ax, ay)) for the closest zombie,
        or None if no zombies.
        """
        rows = typemasks.shape[0]

        # zombies = typemask[:,0] == 1, excluding row 0
        zombie_mask = np.zeros(rows, dtype=bool)
        zombie_mask[1:] = typemasks[1:, 0] > 0.5
        zombie_indices = np.where(zombie_mask)[0]

        print(f"[A* 2D {role.upper()}] zombie rows = {list(zombie_indices)}")

        if zombie_indices.size == 0:
            return None

        # Log each zombie
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
