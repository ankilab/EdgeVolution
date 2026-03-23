"""
Regularized Evolution search strategy.

Based on Real et al. (2019) "Regularized Evolution for Image Classifier
Architecture Search". Maintains a population queue with aging: the oldest
member is removed when a new one is added.
"""

from collections import deque
from typing import List
from coolname import generate_slug
from omegaconf import DictConfig

from .base import SearchStrategy, Candidate, EvaluationResult
from ..src.genepool import GenePool

import numpy as np


class RegularizedEvolutionStrategy(SearchStrategy):
    """Regularized Evolution: tournament selection + aging."""

    def __init__(self, cfg: DictConfig, search_space_registry=None):
        self.cfg = cfg
        self.gene_pool = GenePool(cfg)
        self.search_space_registry = search_space_registry

        self.tournament_size = cfg.search_strategy.get("tournament_size", 25)
        self._generation = 0
        self._population: deque = deque()  # deque of EvaluationResult
        self._all_results: List[EvaluationResult] = []
        self._is_first_generation = True

    @property
    def name(self) -> str:
        return "regularized_evolution"

    def on_generation_start(self, generation: int) -> None:
        self._generation = generation

    def ask(self, n_candidates: int) -> List[Candidate]:
        if self._is_first_generation:
            return self._ask_initial(n_candidates)
        return self._ask_evolved(n_candidates)

    def tell(self, results: List[EvaluationResult]) -> None:
        for result in results:
            self._population.append(result)
            self._all_results.append(result)

        # Apply aging: keep population at a maximum size
        max_pop = self.cfg.hyperparameters.population_size.value * 5
        while len(self._population) > max_pop:
            self._population.popleft()

        self._is_first_generation = False

    def get_best(self, n: int = 1) -> List[EvaluationResult]:
        sorted_results = sorted(self._all_results, key=lambda r: r.fitness, reverse=True)
        return sorted_results[:n]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ask_initial(self, n_candidates):
        """First generation: random chromosomes."""
        candidates = []
        names = self._generate_unique_names(n_candidates)
        for name in names:
            chromosome = self.gene_pool.create_gene_sequence()
            candidates.append(Candidate(name=name, chromosome=chromosome))
        return candidates

    def _ask_evolved(self, n_candidates):
        """Tournament selection on population, then mutate winner."""
        candidates = []
        names = self._generate_unique_names(n_candidates)

        for name in names:
            # Tournament selection: sample k, pick best
            k = min(self.tournament_size, len(self._population))
            sample_indices = np.random.choice(len(self._population), size=k, replace=False)
            sample = [self._population[i] for i in sample_indices]
            winner = max(sample, key=lambda r: r.fitness)

            # Mutate the winner's chromosome
            mutated = self.gene_pool.mutate_chromosome(list(winner.chromosome))
            candidates.append(Candidate(name=name, chromosome=mutated))

        return candidates

    def _generate_unique_names(self, n):
        names = set()
        while len(names) < n:
            random_name = generate_slug(2).replace("-", "_") + f"_{self._generation}"
            if random_name not in names:
                names.add(random_name)
        return sorted(names)
