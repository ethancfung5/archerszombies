# controllers/bfs_controller.py

import numpy as np
from .base_controller import BaseController


class BreadthFirstStubController(BaseController):
    """
    BFS-ish controller that CHASES the closest zombie using full 2D geometry.

    - Knights and archers both:
        * find the closest zombie,
        * rotate sharply to face it,
        * move forward/backward based on heading alignment.
    - The ONLY difference:
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

    name = "BFS (zombie chase 2D)"

    # ------------------------- MAIN ENTRYPOINT -------------------------

    def __call__(self, obs, action_space, agent, t):
        arr = np.asarray(obs, dtype=float)

        if agent.startswith("knight_"):
            return self._chase_and_attack(arr, action_space, agent, t, role="knight")
        elif agent.startswith("archer_"):
            return self._chase_and_attack(arr, action_space, agent, t, role="archer")
        else:
            # Anything else (zombies, arrows, etc.) – just random
            return action_space.sample()

    # ------------------------- SHARED LOGIC ---------------------------

    def _chase_and_attack(self, arr, action_space, agent, t, role: str):
        """
        Shared chasing logic for knights and archers.

        role == "knight"  -> short ATTACK_DIST (melee) + looser aim
        role == "archer"  -> longer ATTACK_DIST (ranged) + tighter aim
        """

        print("\n" + "=" * 70)
        print(f"[BFS 2D {role.upper()}] t = {t}, agent = {agent}")
        print(f"[BFS 2D {role.upper()}] obs shape = {arr.shape}")
        print("-" * 70)

        # Basic sanity
        if arr.ndim != 2:
            print(f"[BFS 2D {role.upper()}] obs ndim != 2, shape={arr.shape}; returning NOOP")
            return self._noop_or_default(action_space)

        rows, cols = arr.shape
        if cols < 11:
            print(f"[BFS 2D {role.upper()}] cols={cols} < 11; not vector+typemask; returning NOOP")
            return self._noop_or_default(action_space)

        # Slice components: [typemask(6), dist, rel_x, rel_y, ang_x, ang_y]
        typemasks = arr[:, :6]
        dists = arr[:, 6]
        rel = arr[:, 7:9]   # (rel_x, rel_y)
        ang = arr[:, 9:11]  # (ang_x, ang_y)

        # Log first few rows for context
        max_rows_to_print = min(rows, 5)
        print(f"[BFS 2D {role.upper()}] first {max_rows_to_print} rows of obs:")
        for i in range(max_rows_to_print):
            print(f"  row {i}: {arr[i]}")

        # Type counts (rough)
        type_names = ["zombie", "archer", "knight", "sword", "arrow", "self"]
        counts = typemasks.sum(axis=0)
        print(f"[BFS 2D {role.upper()}] typemask counts (approx):")
        for name, cnt in zip(type_names, counts):
            print(f"  {name:7s}: {cnt:.1f}")

        # zombies = typemask[:,0] == 1, excluding row 0
        zombie_mask = np.zeros(rows, dtype=bool)
        zombie_mask[1:] = typemasks[1:, 0] > 0.5
        zombie_indices = np.where(zombie_mask)[0]

        print(f"[BFS 2D {role.upper()}] zombie rows = {list(zombie_indices)}")

        if zombie_indices.size == 0:
            print(f"[BFS 2D {role.upper()}] no zombies visible -> NOOP")
            return self._noop_or_default(action_space)

        # Log each zombie
        for idx in zombie_indices:
            dist_i = dists[idx]
            rx_i, ry_i = rel[idx]
            ax_i, ay_i = ang[idx]
            print(
                f"    zombie row={idx}: dist={dist_i:.3f}, "
                f"rel=({rx_i:.3f},{ry_i:.3f}), ang=({ax_i:.3f},{ay_i:.3f})"
            )

        # Pick closest zombie by distance
        closest_idx = zombie_indices[np.argmin(dists[zombie_indices])]
        dist = dists[closest_idx]
        rx, ry = rel[closest_idx]
        zx_ang_x, zx_ang_y = ang[closest_idx]

        print(
            f"[BFS 2D {role.upper()}] CLOSEST zombie row={closest_idx}, "
            f"dist={dist:.3f}, rel=({rx:.3f},{ry:.3f}), ang=({zx_ang_x:.3f},{zx_ang_y:.3f})"
        )

        # Also log row 0 (self)
        self_typemask = typemasks[0]
        self_pos_x, self_pos_y = rel[0]    # in KAZ, row 0 rel usually stores pos
        self_heading_x, self_heading_y = ang[0]
        print(f"[BFS 2D {role.upper()}] row 0 (self) breakdown:")
        print(f"  typemask = {self_typemask}")
        print(f"  pos      = ({self_pos_x:.3f},{self_pos_y:.3f})")
        print(f"  heading  = ({self_heading_x:.3f},{self_heading_y:.3f})")

        # ------------------------ GEOMETRY DECISION -------------------------

        # Different attack ranges & aim cones per role
        if role == "knight":
            ATTACK_DIST = 0.06   # close melee
            AIM_DOT = 0.90       # must be mostly in front
        else:  # archer
            ATTACK_DIST = 0.6 # farther ranged shot
            AIM_DOT = 0.995       # must be tightly in front

        # Shared turning thresholds
        FACING_FWD = 0.99       # very tight forward cone
        FACING_BACK = -0.99     # very tight backward cone

        # Heading vector
        hx, hy = self_heading_x, self_heading_y
        h_norm = np.hypot(hx, hy)
        if h_norm < 1e-6:
            hx, hy = 0.0, -1.0
        else:
            hx, hy = hx / h_norm, hy / h_norm

        # Vector from agent to zombie
        vx, vy = rx, ry
        v_norm = np.hypot(vx, vy)
        if v_norm < 1e-6:
            print(f"[BFS 2D {role.upper()}] virtually on top of zombie -> ATTACK (4)")
            action = 4
            print(f"[BFS 2D {role.upper()}] DECISION: ATTACK, action={action}")
            print("=" * 70)
            return action

        vx, vy = vx / v_norm, vy / v_norm

        # Dot and cross
        dot = hx * vx + hy * vy
        cross_z = hx * vy - hy * vx

        print(f"[BFS 2D {role.upper()}] dot(h,v) = {dot:.3f}, cross_z = {cross_z:.3f}, dist = {dist:.3f}")

        # Attack ONLY if:
        #   - within attack distance for this role
        #   - AND zombie is inside the heading cone (dot > AIM_DOT)
        if dist < ATTACK_DIST and dot > AIM_DOT:
            action = 4  # attack
            reason = f"ATTACK (dist={dist:.3f} < {ATTACK_DIST:.2f} and dot={dot:.3f} > AIM_DOT={AIM_DOT:.2f})"
        else:
            # Not in "attack + heading" cone -> move/rotate
            if dot > FACING_FWD:
                # Facing the zombie well enough -> step forward/back based on distance
                # (simpler: always forward; you can fancy this up later if needed)
                action = 0
                reason = "FORWARD (dot > FACING_FWD)"
            elif dot < FACING_BACK:
                action = 1
                reason = "BACKWARD (dot < FACING_BACK)"
            else:
                # Rotate aggressively toward zombie
                if cross_z < 0:
                    action = 2
                    reason = "ROTATE CCW (cross_z < 0)"
                elif cross_z > 0:
                    action = 3
                    reason = "ROTATE CW (cross_z > 0)"
                else:
                    action = self._noop_or_default(action_space)
                    reason = "NOOP (cross_z == 0, ambiguous side)"

        print(f"[BFS 2D {role.upper()}] DECISION: {reason}, action={action}")
        print("=" * 70)
        return action

    # ------------------------------------------------------------------
    # Helper to choose NOOP if available
    # ------------------------------------------------------------------
    def _noop_or_default(self, action_space):
        """
        Try to return NOOP (5) if action_space supports it; otherwise 0.
        """
        noop = 5
        fallback = 0

        if hasattr(action_space, "n"):
            if noop < action_space.n:
                return noop
            return fallback
        return noop
