"""
Search strategy factory and exports.

Usage:
    from neural_architecture_search.strategies import create_strategy
    strategy = create_strategy(cfg, search_space_registry)
"""

from .base import SearchStrategy, Candidate, EvaluationResult
from .ga_strategy import GAStrategy
from .random_strategy import RandomSearchStrategy
from .regularized_evolution_strategy import RegularizedEvolutionStrategy


def create_strategy(cfg, search_space_registry=None) -> SearchStrategy:
    """Create a search strategy based on the config.

    Args:
        cfg: Hydra DictConfig with a search_strategy.name field.
        search_space_registry: Optional SearchSpaceRegistry instance
            (required for pymoo and ax strategies).

    Returns:
        A SearchStrategy instance.
    """
    name = cfg.search_strategy.name

    if name == "genetic_algorithm":
        return GAStrategy(cfg, search_space_registry)
    elif name == "regularized_evolution":
        return RegularizedEvolutionStrategy(cfg, search_space_registry)
    elif name == "pymoo":
        from .pymoo_strategy import PyMOOStrategy
        return PyMOOStrategy(cfg, search_space_registry)
    elif name == "ax":
        from .ax_strategy import AxStrategy
        return AxStrategy(cfg, search_space_registry)
    elif name == "random":
        return RandomSearchStrategy(cfg, search_space_registry)
    else:
        raise ValueError(
            f"Unknown search strategy: '{name}'. "
            f"Available: genetic_algorithm, regularized_evolution, pymoo, ax, random"
        )
