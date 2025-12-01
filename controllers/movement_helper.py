# controllers/movement_helper.py

import numpy as np


class MovementHelper:
    """
    Shared movement + attack geometry for KAZ controllers.

    Exposes:
      - role_params(role)
      - noop_or_default(action_space)
      - steer_towards_target(...)
    """

    @staticmethod
    def role_params(role: str):
        """
        Return (ATTACK_DIST, AIM_DOT) for the given role.
        """
        if role == "knight":
            # close melee, looser aim cone
            return 0.06, 0.90
        else:
            # archer: farther ranged shot, tight aim cone
            return 0.6, 0.995

    @staticmethod
    def noop_or_default(action_space, noop: int = 5, fallback: int = 0) -> int:
        """
        Try to return NOOP (5) if action_space supports it; otherwise fallback (0).
        """
        if hasattr(action_space, "n") and noop < action_space.n:
            return noop
        return fallback

    @staticmethod
    def steer_towards_target(
        role: str,
        dist: float,
        target_rel,
        self_heading,
        action_space,
        label: str = "CTRL",
    ) -> int:
        """
        Core movement/attack logic given:
          - role: "knight" or "archer"
          - dist: scalar distance to the *true* target (e.g., zombie)
          - target_rel: (rx, ry) vector from self -> local waypoint/target in local coords
          - self_heading: (hx, hy) heading vector of self
          - action_space: PettingZoo-style action space
          - label: prefix used in debug prints (e.g., "BFS 2D KNIGHT")

        Returns: action in {0,1,2,3,4,5}
        """
        rx, ry = target_rel
        hx, hy = self_heading

        ATTACK_DIST, AIM_DOT = MovementHelper.role_params(role)

        # Shared turning thresholds
        FACING_FWD = 0.99       # very tight forward cone
        FACING_BACK = -0.99     # very tight backward cone

        # Normalize heading
        h_norm = np.hypot(hx, hy)
        if h_norm < 1e-6:
            hx, hy = 0.0, -1.0
        else:
            hx, hy = hx / h_norm, hy / h_norm

        # Normalize target direction
        vx, vy = rx, ry
        v_norm = np.hypot(vx, vy)
        if v_norm < 1e-6:
            print(f"[{label}] virtually on top of target -> ATTACK (4)")
            print(f"[{label}] DECISION: ATTACK, action=4")
            return 4

        vx, vy = vx / v_norm, vy / v_norm

        # Dot and cross
        dot = hx * vx + hy * vy
        cross_z = hx * vy - hy * vx

        print(
            f"[{label}] dot(h,v) = {dot:.3f}, "
            f"cross_z = {cross_z:.3f}, dist = {dist:.3f}"
        )

        # Attack ONLY if:
        #   - within attack distance for this role
        #   - AND target is inside the heading cone (dot > AIM_DOT)
        if dist < ATTACK_DIST and dot > AIM_DOT:
            action = 4  # attack
            reason = (
                f"ATTACK (dist={dist:.3f} < {ATTACK_DIST:.2f} and "
                f"dot={dot:.3f} > AIM_DOT={AIM_DOT:.2f})"
            )
        else:
            # Not in "attack + heading" cone -> move/rotate
            if dot > FACING_FWD:
                action = 0
                reason = "FORWARD (dot > FACING_FWD)"
            elif dot < FACING_BACK:
                action = 1
                reason = "BACKWARD (dot < FACING_BACK)"
            else:
                # Rotate aggressively toward target
                if cross_z < 0:
                    action = 2
                    reason = "ROTATE CCW (cross_z < 0)"
                elif cross_z > 0:
                    action = 3
                    reason = "ROTATE CW (cross_z > 0)"
                else:
                    action = MovementHelper.noop_or_default(action_space)
                    reason = "NOOP (cross_z == 0, ambiguous side)"

        print(f"[{label}] DECISION: {reason}, action={action}")
        return action
