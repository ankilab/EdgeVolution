"""Tests for PyMOO integration with Neural Architecture Search."""

import unittest

import numpy as np

from neural_architecture_search.src.search_space_registry import (
    SearchSpaceRegistry,
    ParameterSpec,
    LayerSpec,
)

try:
    from neural_architecture_search.examples.pymoo_integration import (
        NASProblem,
        ArchitectureSampling,
        ArchitectureRepair,
    )
    PYMOO_AVAILABLE = True
except (ImportError, NameError):
    PYMOO_AVAILABLE = False


def _make_small_registry():
    """Create a small test registry with 3 layers: CONV -> POOL -> DENSE."""
    layers = {
        "CONV": LayerSpec(
            name="CONV",
            f_name="Conv2D",
            category="feature",
            params={
                "filters": ParameterSpec(
                    name="filters",
                    param_type="discrete",
                    values=[8, 16, 32, 64],
                ),
            },
            successors=["CONV", "POOL"],
        ),
        "POOL": LayerSpec(
            name="POOL",
            f_name="GlobalAveragePooling2D",
            category="pooling",
            params={},
            successors=["DENSE"],
        ),
        "DENSE": LayerSpec(
            name="DENSE",
            f_name="Dense",
            category="classification",
            params={
                "units": ParameterSpec(
                    name="units",
                    param_type="discrete",
                    values=[32, 64, 128],
                ),
            },
            terminal=True,
        ),
    }

    categories = {
        "feature": {"successors": ["feature", "pooling"]},
        "pooling": {"successors": ["classification"]},
        "classification": {"successors": ["classification"], "terminal": True},
    }

    return SearchSpaceRegistry(
        layers=layers,
        categories=categories,
        start_layers=["CONV"],
        max_layers=8,
    )


@unittest.skipUnless(PYMOO_AVAILABLE, "pymoo is not installed")
class TestNASProblem(unittest.TestCase):
    """Tests for the NASProblem class."""

    def setUp(self):
        """Set up a small registry and NASProblem for each test."""
        self.registry = _make_small_registry()
        self.problem = NASProblem(self.registry)

    def test_initialization(self):
        """NASProblem n_var equals registry.vector_size and n_obj is 3."""
        self.assertEqual(self.problem.n_var, self.registry.vector_size)
        self.assertEqual(self.problem.n_obj, 3)

    def test_dummy_evaluator(self):
        """_dummy_evaluator returns a tuple of 3 numeric values."""
        chromosome = self.registry.create_random_chromosome()
        result = self.problem._dummy_evaluator(chromosome)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertIsInstance(value, (int, float, np.integer, np.floating))

    def test_evaluate_shape(self):
        """_evaluate produces out['F'] with shape (pop_size, 3)."""
        pop_size = 5
        n_var = self.registry.vector_size
        X = np.random.rand(pop_size, n_var).astype(np.float32)
        out = {}

        self.problem._evaluate(X, out)

        self.assertIn("F", out)
        self.assertEqual(out["F"].shape, (pop_size, 3))


@unittest.skipUnless(PYMOO_AVAILABLE, "pymoo is not installed")
class TestArchitectureSampling(unittest.TestCase):
    """Tests for the ArchitectureSampling class."""

    def test_sampling_shape_and_range(self):
        """_do returns array of shape (n_samples, n_var) with values in [0, 1]."""
        registry = _make_small_registry()
        problem = NASProblem(registry)
        sampling = ArchitectureSampling(registry)

        n_samples = 10
        X = sampling._do(problem, n_samples)

        self.assertEqual(X.shape, (n_samples, registry.vector_size))
        self.assertTrue(
            np.all(X >= 0.0),
            "All sampled values must be >= 0.0",
        )
        self.assertTrue(
            np.all(X <= 1.0),
            "All sampled values must be <= 1.0",
        )


@unittest.skipUnless(PYMOO_AVAILABLE, "pymoo is not installed")
class TestArchitectureRepair(unittest.TestCase):
    """Tests for the ArchitectureRepair class."""

    def test_repair_produces_valid_chromosomes(self):
        """After repair, decoded chromosomes obey successor rules."""
        registry = _make_small_registry()
        problem = NASProblem(registry)
        repair = ArchitectureRepair(registry)

        n_individuals = 10
        n_var = registry.vector_size
        X = np.random.rand(n_individuals, n_var).astype(np.float32)

        X_repaired = repair._do(problem, X)

        for i in range(n_individuals):
            chromosome = registry.decode(X_repaired[i], enforce_rules=True)

            if len(chromosome) == 0:
                continue

            # First layer must be a valid start layer
            self.assertIn(
                chromosome[0]["layer"],
                registry.get_start_layers(),
                f"Individual {i}: first layer '{chromosome[0]['layer']}' "
                f"is not a valid start layer",
            )

            # Each subsequent layer must be a valid successor of its predecessor
            for j in range(1, len(chromosome)):
                prev_layer = chromosome[j - 1]["layer"]
                curr_layer = chromosome[j]["layer"]
                valid_successors = registry.get_successors(prev_layer)
                self.assertIn(
                    curr_layer,
                    valid_successors,
                    f"Individual {i}: layer '{curr_layer}' at position {j} "
                    f"is not a valid successor of '{prev_layer}'. "
                    f"Valid successors: {valid_successors}",
                )


if __name__ == "__main__":
    unittest.main()
