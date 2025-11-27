from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------- Basic types --------------------------------------

@dataclass(frozen=True)
class Pos:
    x: float
    y: float


class Direction(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    TURN_CCW = 2
    TURN_CW = 3
    ATTACK = 4
    IDLE = 5



@dataclass(frozen=True)
class ZombieInfo:
    dist: float
    dx: float
    dy: float

NUM_ARCHERS = 0
NUM_KNIGHTS = 2
NUM_SWORDS = NUM_KNIGHTS
MAX_ARROWS = 10
MAX_ZOMBIES = 10

def _as_array(obs) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vector obs, got shape {arr.shape}")
    return arr


def get_agent_heading(obs) -> Tuple[float, float]:
    arr = _as_array(obs)
    if arr.shape[1] != 5:
        raise ValueError(f"Expected width 5, got {arr.shape[1]}")

    row0 = arr[0]
    dir_x = float(row0[3])
    dir_y = float(row0[4])
    norm = np.hypot(dir_x, dir_y)
    if norm < 1e-6:
        return 0.0, -1.0
    return dir_x / norm, dir_y / norm

def get_zombie_vectors(obs) -> List[ZombieInfo]:
    arr = _as_array(obs)
    
    start = 1 + NUM_ARCHERS + NUM_KNIGHTS + NUM_SWORDS + MAX_ARROWS
    end = start + MAX_ZOMBIES
    zombies_block = arr[start:end]

    if zombies_block.size == 0:
        return []

    dists = zombies_block[:, 0]
    mask = dists > 0.0
    zombies_block = zombies_block[mask]

    zombies: List[ZombieInfo] = []
    for row in zombies_block:
        dist, dx, dy, _, _ = row
        zombies.append(ZombieInfo(float(dist), float(dx), float(dy)))
    return zombies


def get_closest_zombie(obs) -> Optional[ZombieInfo]:
    zs = get_zombie_vectors(obs)
    if not zs:
        return None
    return min(zs, key=lambda z: z.dist)


# Action mappings
def action_for_direction(direction: Direction, action_space) -> int:
    idx = int(direction)
    n = getattr(action_space, "n", None)
    if n is not None and n > 0:
        return idx % n
    return idx

def action_attack(action_space) -> int:
    return action_for_direction(Direction.ATTACK, action_space)

def action_idle(action_space) -> int:
    return action_for_direction(Direction.IDLE, action_space)


def decide_knight_action(
    obs,
    action_space,
    zombie: Optional[ZombieInfo],
    attack_dist: float = 0.05,
    t: int = 0,
):

    if zombie is None:
        return action_for_direction(Direction.TURN_CW, action_space), "NO_ZOMBIE_SCAN"

    hx, hy = get_agent_heading(obs)
    zx = float(zombie.dx)
    zy = float(zombie.dy)
    dist = float(zombie.dist) if zombie.dist is not None else np.hypot(zx, zy)
    
    if dist < 1e-6:
        return action_for_direction(Direction.ATTACK, action_space), "ATTACK_overlap"

    zx /= dist
    zy /= dist

    dot = hx * zx + hy * zy
    cross = hy * zx - hx * zy

    if dist < attack_dist:
        if dot > 0.5:
            return action_for_direction(Direction.ATTACK, action_space), "ATTACK_strike"
        else:
            if cross > 0:
                return action_for_direction(Direction.TURN_CCW, action_space), "ATTACK_position_L"
            else:
                return action_for_direction(Direction.TURN_CW, action_space), "ATTACK_position_R"
    
    if dist < attack_dist * 3:
        if dot < 0.8:
            if cross > 0:
                return action_for_direction(Direction.TURN_CCW, action_space), "ATTACK_prep_L"
            else:
                return action_for_direction(Direction.TURN_CW, action_space), "ATTACK_prep_R"
    
    if dot > 0.85:
        return action_for_direction(Direction.FORWARD, action_space), "CHARGE_straight"
    
    elif dot > 0.7:
        if abs(cross) > 0.1 and t % 6 == 0:
            if cross > 0:
                return action_for_direction(Direction.TURN_CCW, action_space), "MINOR_adjust_left"
            else:
                return action_for_direction(Direction.TURN_CW, action_space), "MINOR_adjust_right"
        else:
            return action_for_direction(Direction.FORWARD, action_space), "CHARGE_good_angle"
    
    elif dot > 0.3:
        if t % 3 != 2:
            if cross > 0:
                return action_for_direction(Direction.TURN_CCW, action_space), "ALIGN_turn_left"
            else:
                return action_for_direction(Direction.TURN_CW, action_space), "ALIGN_turn_right"
        else:
            return action_for_direction(Direction.FORWARD, action_space), "CRAWL_forward"
    
    else:
        if cross > 0:
            return action_for_direction(Direction.TURN_CCW, action_space), "FAST_turn_left"
        else:
            return action_for_direction(Direction.TURN_CW, action_space), "FAST_turn_right"


def decide_knight_action_instant(
    obs,
    action_space, 
    zombie: Optional[ZombieInfo],
    attack_dist: float = 0.05,
    t: int = 0,
):
    
    if zombie is None:
        return action_for_direction(Direction.TURN_CW, action_space), "SCAN"

    hx, hy = get_agent_heading(obs)
    zx, zy = float(zombie.dx), float(zombie.dy)
    dist = np.hypot(zx, zy)
    
    if dist < 1e-6:
        return action_for_direction(Direction.ATTACK, action_space), "ATTACK_overlap"
    
    zx /= dist
    zy /= dist
    
    dot = hx * zx + hy * zy
    
    if dist < attack_dist and dot > 0.7:
        return action_for_direction(Direction.ATTACK, action_space), "ATTACK_strike"
    
    if dist < attack_dist * 2:
        if dot < 0.9:
            cross = hy * zx - hx * zy
            if cross > 0:
                return action_for_direction(Direction.TURN_CCW, action_space), "ATTACK_align_L"
            else:
                return action_for_direction(Direction.TURN_CW, action_space), "ATTACK_align_R"
        else:

            return action_for_direction(Direction.FORWARD, action_space), "ATTACK_approach"

    if dot > 0.9:
        return action_for_direction(Direction.FORWARD, action_space), "CHARGE"
    
    cross = hy * zx - hx * zy
    if cross > 0:
        return action_for_direction(Direction.TURN_CCW, action_space), "TURN_L"  
    else:
        return action_for_direction(Direction.TURN_CW, action_space), "TURN_R"


def get_priority_zombie(obs) -> Optional[ZombieInfo]:
    zombies = get_zombie_vectors(obs)
    if not zombies:
        return None

    def priority_score(z: ZombieInfo) -> float:
        distance_weight = z.dist
        escape_weight = -z.dx * 0.3
        return distance_weight - escape_weight
    
    return min(zombies, key=priority_score)

def debug_print_world(obs, header: str = ""):
    if header:
        print(header)

    arr = _as_array(obs)
    print(f"  obs shape: {arr.shape}")
    print("  first 3 rows:")
    for i, row in enumerate(arr[:3]):
        print(f"    {i}: {row}")

    try:
        hx, hy = get_agent_heading(obs)
        print(f"  heading: hx={hx:.3f}, hy={hy:.3f}")
    except Exception as e:
        print(f"  heading parse error: {e}")

    zs = get_zombie_vectors(obs)
    print(f"  zombies visible: {len(zs)}")
    for i, z in enumerate(zs[:3]):
        print(f"    z{i}: dist={z.dist:.3f}, dx={z.dx:.3f}, dy={z.dy:.3f}")