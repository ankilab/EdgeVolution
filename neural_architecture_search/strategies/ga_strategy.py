"""
Genetic Algorithm search strategy.

Wraps the existing GenePool for crossover/mutation, implementing the
ask/tell interface expected by the search loop.
"""

from typing import List
from coolname import generate_slug
from omegaconf import DictConfig

from .base import SearchStrategy, Candidate, EvaluationResult
from ..src.genepool import GenePool


class GAStrategy(SearchStrategy):
    """Standard Genetic Algorithm: selection, crossover, mutation."""

    def __init__(self, cfg: DictConfig, search_space_registry=None):
        self.cfg = cfg
        self.gene_pool = GenePool(cfg)
        self.search_space_registry = search_space_registry

        self._generation = 0
        self._all_results: List[EvaluationResult] = []
        self._fittest: dict = {}  # name -> chromosome for crossover
        self._is_first_generation = True

    @property
    def name(self) -> str:
        return "genetic_algorithm"

    def on_generation_start(self, generation: int) -> None:
        self._generation = generation
        self._apply_decay_schedules()

    def ask(self, n_candidates: int) -> List[Candidate]:
        if self._is_first_generation:
            return self._ask_initial(n_candidates)
        return self._ask_evolved(n_candidates)

    def tell(self, results: List[EvaluationResult]) -> None:
        # Sort by fitness descending
        results_sorted = sorted(results, key=lambda r: r.fitness, reverse=True)

        # Keep top K for crossover
        k = self.cfg.hyperparameters.num_best_models_crossover.value
        top = results_sorted[:k]

        self._fittest = {r.name: r.chromosome for r in top}
        self._all_results.extend(results_sorted)
        self._is_first_generation = False

    def get_best(self, n: int = 1) -> List[EvaluationResult]:
        sorted_results = sorted(self._all_results, key=lambda r: r.fitness, reverse=True)
        return sorted_results[:n]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ask_initial(self, n_candidates):
        """First generation: create random chromosomes."""
        candidates = []
        names = self._generate_unique_names(n_candidates)
        for name in names:
            chromosome = self.gene_pool.create_gene_sequence()
            candidates.append(Candidate(name=name, chromosome=chromosome))
        return candidates

    def _ask_evolved(self, n_candidates):
        """Subsequent generations: crossover + mutation from fittest."""
        # Crossover from in-memory chromosomes
        new_chromosomes, parents = self.gene_pool.crossover_from_chromosomes(self._fittest)

        # Generate new names
        names = self._generate_unique_names(len(new_chromosomes))

        # Mutate each chromosome
        candidates = []
        for name, chromosome in zip(names, new_chromosomes):
            mutated = self.gene_pool.mutate_chromosome(chromosome)
            candidates.append(Candidate(name=name, chromosome=mutated))
        return candidates

    def _generate_unique_names(self, n):
        """Generate n unique random names."""
        names = set()
        while len(names) < n:
            random_name = generate_slug(2).replace("-", "_") + f"_{self._generation}"
            if random_name not in names:
                names.add(random_name)
        return sorted(names)

    def _apply_decay_schedules(self):
        """Apply decay schedules for population size, crossover count, mutation rate.

        Note: population_size decay is also applied centrally in main._run_search()
        so that non-GA strategies get the correct value too.
        """
        gen = self._generation

        decay = self.cfg.hyperparameters.population_size_decay.value
        self.cfg.hyperparameters.population_size.value = next(
            s[1] for s in decay[::-1] if gen >= s[0]
        )

        decay = self.cfg.hyperparameters.num_best_models_crossover_decay.value
        self.cfg.hyperparameters.num_best_models_crossover.value = next(
            s[1] for s in decay[::-1] if gen >= s[0]
        )

        decay = self.cfg.hyperparameters.mutation_rate_decay.value
        self.cfg.hyperparameters.mutation_rate.value = next(
            s[1] for s in decay[::-1] if gen >= s[0]
        )
