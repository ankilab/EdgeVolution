"""
PyMOO multi-objective search strategy (NSGA-II / NSGA-III / MOEA/D).

Requires: pip install pymoo

Uses the SearchSpaceRegistry for encoding/decoding between PyMOO's
continuous vector representation and EdgeVolution's chromosome format.
"""

from typing import List
from coolname import generate_slug
from omegaconf import DictConfig

import numpy as np

from .base import SearchStrategy, Candidate, EvaluationResult

try:
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.core.sampling import Sampling
    from pymoo.core.population import Population
    from pymoo.core.individual import Individual
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


class PyMOOStrategy(SearchStrategy):
    """Multi-objective optimization using PyMOO (NSGA-II by default)."""

    def __init__(self, cfg: DictConfig, search_space_registry):
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "PyMOO is required for the 'pymoo' search strategy. "
                "Install with: pip install pymoo"
            )
        if search_space_registry is None:
            raise ValueError(
                "PyMOO strategy requires a SearchSpaceRegistry. "
                "Provide a +search_space= config."
            )

        self.cfg = cfg
        self.registry = search_space_registry

        self._generation = 0
        self._all_results: List[EvaluationResult] = []
        self._pending_vectors: np.ndarray = None  # vectors for current ask
        self._is_first_generation = True

        # Build PyMOO algorithm
        algo_name = cfg.search_strategy.get("algorithm", "nsga2")
        # population_size.value may still be 0 (placeholder) at init time;
        # resolve from the decay schedule to get the initial population size.
        pop_size = cfg.hyperparameters.population_size.value
        if not pop_size:
            decay = cfg.hyperparameters.population_size_decay.value
            pop_size = decay[0][1] if decay else 20
        crossover_prob = cfg.search_strategy.get("crossover_prob", 0.9)
        mutation_eta = cfg.search_strategy.get("mutation_eta", 20)

        sampling = _RegistrySampling(self.registry)
        repair = _RegistryRepair(self.registry)

        if algo_name == "nsga3":
            n_obj = 2  # accuracy + memory (energy if MCU)
            if cfg.hyperparameters.optimize_for_MCU.value:
                n_obj = 3
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            self.algorithm = NSGA3(
                pop_size=pop_size,
                ref_dirs=ref_dirs,
                sampling=sampling,
                repair=repair,
            )
        else:
            self.algorithm = NSGA2(
                pop_size=pop_size,
                sampling=sampling,
                repair=repair,
            )

        # Build problem
        n_obj = 2
        if cfg.hyperparameters.optimize_for_MCU.value:
            n_obj = 3
        self._problem = _NASProblem(self.registry, n_obj=n_obj)

        # Initialize algorithm
        self.algorithm.setup(self._problem, termination=("n_gen", 99999))

    @property
    def name(self) -> str:
        return "pymoo"

    def on_generation_start(self, generation: int) -> None:
        self._generation = generation

    def ask(self, n_candidates: int) -> List[Candidate]:
        if self._is_first_generation:
            # Let PyMOO initialize population
            pop = self.algorithm.infill()
            self._pending_vectors = pop.get("X")
        else:
            pop = self.algorithm.infill()
            self._pending_vectors = pop.get("X")

        candidates = []
        names = self._generate_unique_names(len(self._pending_vectors))

        for i, (name, vector) in enumerate(zip(names, self._pending_vectors)):
            chromosome = self.registry.decode(vector, enforce_rules=True)
            candidates.append(Candidate(
                name=name,
                chromosome=chromosome,
                metadata={"pymoo_index": i},
            ))

        return candidates

    def tell(self, results: List[EvaluationResult]) -> None:
        self._all_results.extend(results)

        # Build objective matrix for PyMOO (all objectives minimized)
        F = []
        for result in results:
            objectives = [
                -result.val_acc,  # maximize accuracy -> minimize negative
                result.memory_footprint_tflite,
            ]
            if self.cfg.hyperparameters.optimize_for_MCU.value:
                energy = result.energy_consumption if result.energy_consumption else 0.0
                objectives.append(energy)
            F.append(objectives)
        F = np.array(F)

        # Reconstruct PyMOO population with evaluation results
        pop = Population.new("X", self._pending_vectors)
        pop.set("F", F)

        self.algorithm.advance(infills=pop)
        self._is_first_generation = False

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


# ------------------------------------------------------------------
# PyMOO helpers (only defined when pymoo is installed)
# ------------------------------------------------------------------

if PYMOO_AVAILABLE:

    class _NASProblem(Problem):
        """NAS problem definition for PyMOO."""

        def __init__(self, registry, n_obj=2):
            super().__init__(
                n_var=registry.vector_size,
                n_obj=n_obj,
                n_constr=0,
                xl=np.zeros(registry.vector_size),
                xu=np.ones(registry.vector_size),
            )
            self.registry = registry

        def _evaluate(self, X, out, *args, **kwargs):
            # Evaluation is handled externally via ask/tell
            pass

    class _RegistrySampling(Sampling):
        """Generate valid initial architectures using the registry."""

        def __init__(self, registry):
            super().__init__()
            self.registry = registry

        def _do(self, problem, n_samples, **kwargs):
            X = np.zeros((n_samples, problem.n_var), dtype=np.float32)
            for i in range(n_samples):
                chromosome = self.registry.create_random_chromosome()
                X[i] = self.registry.encode(chromosome)
            return X

    class _RegistryRepair(Repair):
        """Repair invalid architectures after crossover/mutation."""

        def __init__(self, registry):
            super().__init__()
            self.registry = registry

        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                chromosome = self.registry.decode(X[i], enforce_rules=True)
                if len(chromosome) > 0:
                    X[i] = self.registry.encode(chromosome)
            return X
