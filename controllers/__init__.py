# controllers/__init__.py

from .random_controller import RandomController
from .greedy_search import GreedyController
from .dfs_controller import DepthFirstSearchController
from .bfs_controller import BreadthFirstStubController
from .astar_controller import AStarController

# Registry to populate the dropdown
CONTROLLERS = [
    RandomController(),
    GreedyController(),
    DepthFirstSearchController(),
    BreadthFirstStubController(),
    AStarController(),
]


def get_controller_by_name(name: str):
    """Helper to look up a controller instance by its display name."""
    return next((c for c in CONTROLLERS if c.name == name), CONTROLLERS[0])
