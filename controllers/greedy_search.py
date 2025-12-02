import numpy as np
from .base_controller import BaseController


class GreedyController(BaseController):
    """
    Greedy controller for Knights & Archers vs Zombies.

    Action mapping:
      0 -> move forward
      1 -> move downward
      2 -> turn CCW
      3 -> turn CW
      4 -> attack
      5 -> do nothing
    """

    name = "Greedy Search"

    # --- Knight thresholds ---
    KNIGHT_ATTACK_DIST   = 0.06   # knight must be very close to attack
    KNIGHT_ALIGN_EPS     = 0.03

    # --- Archer thresholds ---
    ARCHER_TOO_CLOSE     = 0.12   # retreat if closer than this
    ARCHER_OPTIMAL_MAX   = 0.40   # if farther than this, move closer
    ARCHER_ALIGN_EPS     = 0.03   # angular tolerance
    LEAD_FACTOR          = 0.35   # how far ahead to aim

    # --- Shared ---
    MIN_DIST_NOISE       = 0.02   # ignore tiny distances

    def __call__(self, obs, action_space, agent, t):
        obs = np.asarray(obs, dtype=float)

        if obs.ndim != 2 or obs.shape[1] < 11:
            return 5

        typemasks = obs[:, :6]
        dists     = obs[:, 6]
        rel       = obs[:, 7:9]
        heading   = obs[0, 9:11]

        if not np.any(np.isfinite(heading)) or np.allclose(heading, 0.0):
            return 0

        hx, hy = heading

        # -----------------------------
        # 1. Furthest-down zombie
        # -----------------------------

        zombie_mask = np.zeros(len(obs), dtype=bool)
        zombie_mask[1:] = typemasks[1:, 0] > 0.5

        zombie_idxs = np.where(zombie_mask)[0]
        if zombie_idxs.size == 0:
            return 5

        rel_x = rel[:, 0]
        rel_y = rel[:, 1]

        target_idx = zombie_idxs[np.argmax(rel_y[zombie_idxs])]

        dist = dists[target_idx]
        rx, ry = rel_x[target_idx], rel_y[target_idx]

        if not np.any(np.isfinite([rx, ry])) or (abs(rx) < 1e-6 and abs(ry) < 1e-6):
            return 5

        r_norm = float(np.sqrt(rx * rx + ry * ry))
        if r_norm < 1e-6:
            return 5

        # -----------------------------
        # 2. Cross for knight
        # -----------------------------
        cross_knight = hx * ry - hy * rx

        def rotate_step_knight():
            return 3 if cross_knight > 0 else 2

        # -----------------------------
        # 3. Lead vector for archer
        # -----------------------------

        # Aim slightly ahead of the zombie (downwards),
        ry_lead = ry + self.LEAD_FACTOR * dist
        rx_lead = rx
        r_lead_norm = float(np.sqrt(rx_lead * rx_lead + ry_lead * ry_lead))
        if r_lead_norm < 1e-6:
            r_lead_norm = 1.0

        cross_archer = hx * ry_lead - hy * rx_lead

        def rotate_step_archer():
            return 3 if cross_archer > 0 else 2

        # -----------------------------
        # 4. Delegate by agent type
        # -----------------------------
        if agent.startswith("knight"):
            return self._knight_policy(dist, cross_knight, rotate_step_knight)

        if agent.startswith("archer"):
            return self._archer_policy(
                dist,
                cross_archer,
                rotate_step_archer,
            )
        return 5

    # ------------------ Knight ------------------ #
    def _knight_policy(self, dist, cross, rotate_step):
        # Very close -> attack
        if dist < self.KNIGHT_ATTACK_DIST and dist > self.MIN_DIST_NOISE:
            return 4  # attack

        # If misaligned, rotate toward target
        if abs(cross) > self.KNIGHT_ALIGN_EPS:
            return rotate_step()

        # Roughly aligned but not in range -> move forward
        return 0

    # ------------------ Archer ------------------ #
    def _archer_policy(self, dist, cross_lead, rotate_step,):
        # Too close -> retreat
        if dist < self.ARCHER_TOO_CLOSE:
            return 1

        # Too far -> move closer
        if dist > self.ARCHER_OPTIMAL_MAX:
            if abs(cross_lead) > self.ARCHER_ALIGN_EPS:
                return rotate_step()
            else:
                return 0

        # Align and shoot
        if abs(cross_lead) > self.ARCHER_ALIGN_EPS:
            return rotate_step()

        if dist > self.MIN_DIST_NOISE:
            return 4

        return 5
