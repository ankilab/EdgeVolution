"""
Ax/BoTorch Bayesian Optimization search strategy.

Requires: pip install ax-platform

Uses Ax's AxClient API for sequential model-based optimization.
Initial trials use Sobol quasi-random sampling, then switches to
Gaussian Process-based Bayesian optimization.
"""

from typing import List
from coolname import generate_slug
from omegaconf import DictConfig

import numpy as np

from .base import SearchStrategy, Candidate, EvaluationResult

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


class AxStrategy(SearchStrategy):
    """Bayesian Optimization using Ax/BoTorch."""

    def __init__(self, cfg: DictConfig, search_space_registry):
        if not AX_AVAILABLE:
            raise ImportError(
                "Ax is required for the 'ax' search strategy. "
                "Install with: pip install ax-platform"
            )
        if search_space_registry is None:
            raise ValueError(
                "Ax strategy requires a SearchSpaceRegistry. "
                "Provide a +search_space= config."
            )

        self.cfg = cfg
        self.registry = search_space_registry
        self.n_initial_random = cfg.search_strategy.get("n_initial_random", 10)

        self._generation = 0
        self._all_results: List[EvaluationResult] = []
        self._pending_trials: List[tuple] = []  # (trial_index, vector)

        # Build Ax experiment
        self.ax_client = AxClient(verbose_logging=False)

        parameters = []
        for i in range(self.registry.vector_size):
            parameters.append({
                "name": f"x{i}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
            })

        objectives = {"val_acc": ObjectiveProperties(minimize=False)}

        self.ax_client.create_experiment(
            name="edgevolution_nas",
            parameters=parameters,
            objectives=objectives,
        )

    @property
    def name(self) -> str:
        return "ax"

    def on_generation_start(self, generation: int) -> None:
        self._generation = generation

    def ask(self, n_candidates: int) -> List[Candidate]:
        candidates = []
        self._pending_trials = []
        names = self._generate_unique_names(n_candidates)

        for name in names:
            trial_params, trial_index = self.ax_client.get_next_trial()

            # Convert Ax parameter dict to vector
            vector = np.array([trial_params[f"x{i}"] for i in range(self.registry.vector_size)])

            # Decode to chromosome
            chromosome = self.registry.decode(vector, enforce_rules=True)

            self._pending_trials.append((trial_index, vector))
            candidates.append(Candidate(
                name=name,
                chromosome=chromosome,
                metadata={"trial_index": trial_index},
            ))

        return candidates

    def tell(self, results: List[EvaluationResult]) -> None:
        self._all_results.extend(results)

        for result, (trial_index, _) in zip(results, self._pending_trials):
            raw_data = {"val_acc": (result.val_acc, None)}
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=raw_data,
            )

        self._pending_trials = []

    def get_best(self, n: int = 1) -> List[EvaluationResult]:
        sorted_results = sorted(self._all_results, key=lambda r: r.fitness, reverse=True)
        return sorted_results[:n]

    def _generate_unique_names(self, n):
        names = set()
        while len(names) < n:
            random_name = generate_slug(2).replace("-", "_") + f"_{self._generation}"
            if random_name not in names:
                names.add(random_name)
        return sorted(names)
