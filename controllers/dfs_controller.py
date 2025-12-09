# controllers/dfs_controller.py

"""
Depth-first search (DFS) controller for the Knights Archers Zombies (KAZ) environment.

This controller:
- Builds an agent-centric grid around the current agent.
- Uses DFS on the grid to plan a short path toward the closest zombie.
- Converts the first grid step of that path into a continuous waypoint.
- Delegates low-level action selection to MovementHelper.
"""

from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .base_controller import BaseController
from .movement_helper import MovementHelper

# (x,y) integer grid cell
Coord = Tuple[int, int]

class DepthFirstSearchController(BaseController):
    """
    Controller that applies depth-first search on a local grid to chase zombies.

    The agent is treated as being at the origin (0, 0) in a discrete grid.
    At each timestep, the closest zombie is projected into this grid and
    DFS is used to compute a path. Only the first step of that path is
    converted back into continuous coordinates and passed to MovementHelper.
    """
    
    name = "DFS (grid chase 2D)"

    # Grid resolution and planning radius in agent's local frames
    CELL_SIZE = 0.1      # world units per grid cell in local frame
    GRID_RADIUS = 10     # plan on [-R..R] x [-R..R]

    def __call__(self, obs, action_space, agent, t):
        """
        Main entry point used by the environment.

        Depending on the agent type, delegates to a DFS-based policy
        (for knights and archers) or returns a random action (for others).
        """
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._act_with_dfs(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._act_with_dfs(arr, action_space, agent, t, role="archer")
        else:
            # Non-controlled entities (zombies, arrows, etc.)
            return action_space.sample()

    def _act_with_dfs(self, arr, action_space, agent, t, role: str):
        """
        Apply the DFS policy for a given controlled agent (knight or archer).

        Steps:
        - Parse the vector observation.
        - Identify the current agent row and the closest zombie.
        - Run DFS on the local grid to obtain a waypoint.
        - Use MovementHelper to translate the waypoint into an action.
        """
        
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

        # Extract agent-specific information
        self_typemask = typemasks[self_idx]
        self_pos_x, self_pos_y = rel[self_idx]
        self_heading_x, self_heading_y = ang[self_idx]

        print(f"[{label}] self row = {self_idx}")
        print(f"[{label}] self typemask = {self_typemask}")
        print(f"[{label}] self pos      = ({self_pos_x:.3f},{self_pos_y:.3f})")
        print(f"[{label}] self heading  = ({self_heading_x:.3f},{self_heading_y:.3f})")

        # Select closest zombie as the planning target
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

        # Plan using DFS on the local grid
        next_step_rel = self._compute_dfs_step(
            goal_rel=(z_rx, z_ry),
            label=label,
        )
        # If DFS fails, fall back to a direct chase toward the zombie
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

        # Convert waypoint and heading into an actual environment action
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
    #---------------------------------------------------------------------
    # DFS on Local grid 
    #---------------------------------------------------------------------
    def _compute_dfs_step(self,goal_rel: Tuple[float, float],label: str,) -> Optional[Tuple[float, float]]:
        """
        Run DFS on the local grid to get the next waypoint in relative coordinates.

        The goal_rel position is discretized into a grid cell. DFS is then used to
        compute a path from (0, 0) to that goal cell. Only the second cell in the
        path (first step from the origin) is converted back to continuous coordinates.
        """
        gx, gy = goal_rel
        cell_size = self.CELL_SIZE
        R = self.GRID_RADIUS

        # discretize goal into grid coordinates
        goal_ix = int(round(gx / cell_size))
        goal_iy = int(round(gy / cell_size))

        # clamp goal to the planning window
        goal_ix = max(-R, min(R, goal_ix))
        goal_iy = max(-R, min(R, goal_iy))

        start: Coord = (0, 0)
        goal: Coord = (goal_ix, goal_iy)

        print(
            f"[{label}] DFS planning on grid [-{R}..{R}]^2 with cell_size={cell_size}: "
            f"start={start} -> goal={goal}"
        )
        
        # If the projected goal coincides with the origin, use the original vector.
        if start == goal:
            print(f"[{label}] start == goal in grid; using direct target.")
            return goal_rel

        path = self._dfs_grid_search(start, goal, R, label)

        # Need at least one step beyond the start cell
        if not path or len(path) < 2:
            print(f"[{label}] DFS found no usable path: {path}")
            return None

        # Set the next cell after the origin is the planned local waypoint
        next_cell = path[1]
        nx, ny = next_cell

        next_rx = nx * cell_size
        next_ry = ny * cell_size

        print(f"[{label}] DFS path (first few cells): {path[:5]}")
        print(
            f"[{label}] next DFS waypoint in local frame: "
            f"({next_rx:.3f}, {next_ry:.3f})"
        )
        return (next_rx, next_ry)

    def _dfs_grid_search(self, start: Coord,goal: Coord,R: int,label: str,) -> Optional[List[Coord]]:
        """
        Perform depth-first search on a bounded 2D grid.

        The grid is four-connected, with neighbors in {up, down, left, right}.
        Neighbor expansion is ordered using a simple Euclidean-distance heuristic
        so that nodes closer to the goal are explored earlier, while still using
        a stack-based DFS frontier.
        """

        base_neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def in_bounds(x: int, y: int) -> bool:
            return -R <= x <= R and -R <= y <= R

        def heuristic(a: Coord, b: Coord) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx * dx + dy * dy) ** 0.5

        # LIFO stack = DFS frontier
        stack: List[Coord] = [start]
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        visited: Set[Coord] = {start}

        print(f"[{label}][DFS] start={start}, goal={goal}, R={R}")

        while stack:
            current = stack.pop()   # <-- DFS core: pop from stack
            print(f"[{label}][DFS] pop {current} from stack")

            if current == goal:
                print(f"[{label}][DFS] reached goal at {current}")
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

            # Explore neighbors that are closer to the goal first (still DFS via stack)
            candidates.sort(key=lambda x: x[1], reverse=True)

            for neighbor, h in candidates:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)  # <-- DFS: push onto stack
                print(f"[{label}][DFS]   push neighbor={neighbor} (h={h:.3f})")

        print(f"[{label}][DFS] exhausted stack; no path to {goal}")
        return None


    @staticmethod
    def _reconstruct_path(came_from: Dict[Coord, Optional[Coord]],current: Coord,) -> List[Coord]:
        """
        Reconstruct a path from the start node to `current`
        using the `came_from` predecessor map.
        """
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    #----------------------
    # Observation utilities
    #----------------------
    def _parse_vector_obs(self, arr, role):
        """
        Interpret obs as [N, 11]:

            [typemask(6), dist, rel_x, rel_y, ang_x, ang_y]

        Returns:
            (typemasks, dists, rel, ang, self_idx) or None if parsing fails.
        """
        # Handle flat 1D observations by reshaping into rows of length 11
        if arr.ndim == 1:
            if arr.size % 11 == 0:
                arr = arr.reshape(-1, 11)
                print(f"[DFS 2D {role.upper()}] reshaped 1D obs -> {arr.shape}")
            else:
                print(f"[DFS 2D {role.upper()}] 1D obs of size {arr.size} cannot be reshaped to 11 cols")
                return None

        # Expect a matrix with at least 11 columns
        if arr.ndim != 2:
            print(f"[DFS 2D {role.upper()}] obs ndim != 2 after reshape, shape={arr.shape}")
            return None

        rows, cols = arr.shape
        if cols < 11:
            print(f"[DFS 2D {role.upper()}] cols={cols} < 11; cannot parse vector+typemask")
            return None

        # Split into semantic concepts
        typemasks = arr[:, :6]
        dists = arr[:, 6]
        rel = arr[:, 7:9]
        ang = arr[:, 9:11]

        # typemask[:, 5] is used to flag the "self" row
        self_bits = typemasks[:, 5] > 0.5
        self_indices = np.where(self_bits)[0]

        if self_indices.size == 0:
            # If self is not explicitly flagged, fall back to row 0
            self_idx = 0
            print(f"[DFS 2D {role.upper()}] no explicit self row; defaulting self_idx=0")
        else:
            self_idx = int(self_indices[0])
            print(f"[DFS 2D {role.upper()}] detected self_idx={self_idx} via typemask[5]")

        return typemasks, dists, rel, ang, self_idx

    def _debug_log_obs(self, arr, typemasks, role):
        """
        Print a small summary of the current observation for debugging.
        """
        
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
        """
        Identify the closest zombie in the observation, excluding the agent itself.

        Returns:
            (index, distance, (rel_x, rel_y), (ang_x, ang_y)) or None if no zombie is visible.
        """
        rows = typemasks.shape[0]

        # Column 0 in the typemask corresponds to "zombie"
        zombie_mask = typemasks[:, 0] > 0.5
        zombie_mask[self_idx] = False
        zombie_indices = np.where(zombie_mask)[0]

        print(f"[DFS 2D {role.upper()}] zombie rows = {list(zombie_indices)}")

        if zombie_indices.size == 0:
            return None

        # Log basic information for all visible zombies
        for idx in zombie_indices:
            dist_i = dists[idx]
            rx_i, ry_i = rel[idx]
            ax_i, ay_i = ang[idx]
            print(
                f"    zombie row={idx}: dist={dist_i:.3f}, "
                f"rel=({rx_i:.3f},{ry_i:.3f}), ang=({ax_i:.3f},{ay_i:.3f})"
            )

        # Choose the zombie with minimal distance
        closest_idx = zombie_indices[np.argmin(dists[zombie_indices])]
        dist = dists[closest_idx]
        rx, ry = rel[closest_idx]
        zx_ang_x, zx_ang_y = ang[closest_idx]

        return closest_idx, dist, (rx, ry), (zx_ang_x, zx_ang_y)
