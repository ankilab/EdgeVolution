"""
Random Search baseline strategy.

Generates random architectures each generation with no selection,
crossover, or mutation. Useful as a baseline to measure whether
more sophisticated strategies provide meaningful improvement.
"""

from typing import List
from coolname import generate_slug
from omegaconf import DictConfig

from .base import SearchStrategy, Candidate, EvaluationResult
from ..src.genepool import GenePool


class RandomSearchStrategy(SearchStrategy):
    """Random Search: generate random chromosomes every generation."""

    def __init__(self, cfg: DictConfig, search_space_registry=None):
        self.cfg = cfg
        self.gene_pool = GenePool(cfg)
        self.search_space_registry = search_space_registry

        self._generation = 0
        self._all_results: List[EvaluationResult] = []

    @property
    def name(self) -> str:
        return "random"

    def on_generation_start(self, generation: int) -> None:
        self._generation = generation

    def ask(self, n_candidates: int) -> List[Candidate]:
        candidates = []
        names = self._generate_unique_names(n_candidates)
        for name in names:
            chromosome = self.gene_pool.create_gene_sequence()
            candidates.append(Candidate(name=name, chromosome=chromosome))
        return candidates

    def tell(self, results: List[EvaluationResult]) -> None:
        self._all_results.extend(results)

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
