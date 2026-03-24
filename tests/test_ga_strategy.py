"""Tests for GAStrategy: population decay schedules and individual naming."""

import unittest
from unittest.mock import patch, MagicMock

from omegaconf import OmegaConf

from neural_architecture_search.strategies.ga_strategy import GAStrategy
from neural_architecture_search.strategies.base import EvaluationResult


def _make_cfg(
    population_size_decay=None,
    num_best_models_crossover_decay=None,
    mutation_rate_decay=None,
):
    """Build a minimal OmegaConf config for GAStrategy.

    Only the fields that _apply_decay_schedules and _generate_unique_names
    touch are required; GenePool is patched out in tests.
    """
    if population_size_decay is None:
        population_size_decay = [[1, 100], [2, 50], [10, 25]]
    if num_best_models_crossover_decay is None:
        num_best_models_crossover_decay = [[1, 50], [2, 25], [10, 10]]
    if mutation_rate_decay is None:
        mutation_rate_decay = [[1, 30], [2, 25], [5, 20], [10, 15]]

    return OmegaConf.create({
        "hyperparameters": {
            "population_size_decay": {"value": population_size_decay},
            "num_best_models_crossover_decay": {"value": num_best_models_crossover_decay},
            "mutation_rate_decay": {"value": mutation_rate_decay},
            "population_size": {"value": 0},
            "num_best_models_crossover": {"value": 0},
            "mutation_rate": {"value": 0},
            "max_num_feature_layers": {"value": 8},
            "max_num_classification_layers": {"value": 4},
        },
        "search_space": {
            "layers": {
                "C_1D": {
                    "f_name": "get_conv1d_block",
                    "category": "feature_extraction",
                    "filters": [8, 64, 8],
                    "kernel_size": [3, 7, 2],
                },
                "GAP_1D": {
                    "f_name": "GlobalAveragePooling1D",
                    "category": "global_pooling",
                },
                "D": {
                    "f_name": "Dense",
                    "category": "classification",
                    "units": [8, 128, 8],
                },
            },
            "categories": {
                "feature_extraction": {"successors": ["feature_extraction", "global_pooling"]},
                "global_pooling": {"successors": ["classification"]},
                "classification": {"successors": ["classification"], "terminal": True},
            },
            "start": ["C_1D"],
        },
    })


@patch("neural_architecture_search.strategies.ga_strategy.GenePool")
class TestDecaySchedules(unittest.TestCase):
    """Test that _apply_decay_schedules produces the correct values for
    every generation with the default speech_commands config."""

    def _make_strategy(self, cfg, MockGenePool):
        MockGenePool.return_value = MagicMock()
        return GAStrategy(cfg)

    # ------------------------------------------------------------------
    # Population size
    # ------------------------------------------------------------------

    def test_generation_1_population_size(self, MockGenePool):
        """Generation 1 must use the first decay bracket (100)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(1)
        self.assertEqual(cfg.hyperparameters.population_size.value, 100)

    def test_generation_2_population_size(self, MockGenePool):
        """Generation 2 transitions to second bracket (50)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(2)
        self.assertEqual(cfg.hyperparameters.population_size.value, 50)

    def test_generation_9_population_size(self, MockGenePool):
        """Generation 9 is still in the second bracket (50)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(9)
        self.assertEqual(cfg.hyperparameters.population_size.value, 50)

    def test_generation_10_population_size(self, MockGenePool):
        """Generation 10 transitions to third bracket (25)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(10)
        self.assertEqual(cfg.hyperparameters.population_size.value, 25)

    def test_generation_20_population_size(self, MockGenePool):
        """Generation 20 stays in third bracket (25)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(20)
        self.assertEqual(cfg.hyperparameters.population_size.value, 25)

    # ------------------------------------------------------------------
    # Crossover count
    # ------------------------------------------------------------------

    def test_generation_1_crossover_count(self, MockGenePool):
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(1)
        self.assertEqual(cfg.hyperparameters.num_best_models_crossover.value, 50)

    def test_generation_2_crossover_count(self, MockGenePool):
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(2)
        self.assertEqual(cfg.hyperparameters.num_best_models_crossover.value, 25)

    def test_generation_10_crossover_count(self, MockGenePool):
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(10)
        self.assertEqual(cfg.hyperparameters.num_best_models_crossover.value, 10)

    # ------------------------------------------------------------------
    # Mutation rate
    # ------------------------------------------------------------------

    def test_generation_1_mutation_rate(self, MockGenePool):
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(1)
        self.assertEqual(cfg.hyperparameters.mutation_rate.value, 30)

    def test_generation_4_mutation_rate(self, MockGenePool):
        """Generation 4 is still in second bracket (25)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(4)
        self.assertEqual(cfg.hyperparameters.mutation_rate.value, 25)

    def test_generation_5_mutation_rate(self, MockGenePool):
        """Generation 5 transitions to third bracket (20)."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(5)
        self.assertEqual(cfg.hyperparameters.mutation_rate.value, 20)

    def test_generation_10_mutation_rate(self, MockGenePool):
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(10)
        self.assertEqual(cfg.hyperparameters.mutation_rate.value, 15)

    # ------------------------------------------------------------------
    # Full schedule walk-through
    # ------------------------------------------------------------------

    def test_all_generations_population_size(self, MockGenePool):
        """Walk through 20 generations and verify every population_size."""
        expected = {
            1: 100,
            2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 50, 8: 50, 9: 50,
            10: 25, 11: 25, 12: 25, 13: 25, 14: 25, 15: 25, 16: 25,
            17: 25, 18: 25, 19: 25, 20: 25,
        }
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        for gen in range(1, 21):
            strategy.on_generation_start(gen)
            self.assertEqual(
                cfg.hyperparameters.population_size.value,
                expected[gen],
                f"Population size mismatch at generation {gen}",
            )

    def test_all_generations_mutation_rate(self, MockGenePool):
        """Walk through 20 generations and verify every mutation_rate."""
        expected = {
            1: 30,
            2: 25, 3: 25, 4: 25,
            5: 20, 6: 20, 7: 20, 8: 20, 9: 20,
            10: 15, 11: 15, 12: 15, 13: 15, 14: 15, 15: 15, 16: 15,
            17: 15, 18: 15, 19: 15, 20: 15,
        }
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        for gen in range(1, 21):
            strategy.on_generation_start(gen)
            self.assertEqual(
                cfg.hyperparameters.mutation_rate.value,
                expected[gen],
                f"Mutation rate mismatch at generation {gen}",
            )

    # ------------------------------------------------------------------
    # Edge / corner cases
    # ------------------------------------------------------------------

    def test_single_bracket_decay(self, MockGenePool):
        """Decay with only one bracket should use that value everywhere."""
        cfg = _make_cfg(population_size_decay=[[1, 42]])
        strategy = self._make_strategy(cfg, MockGenePool)
        for gen in (1, 5, 10, 100):
            strategy.on_generation_start(gen)
            self.assertEqual(cfg.hyperparameters.population_size.value, 42)

    def test_many_brackets_picks_latest(self, MockGenePool):
        """With many brackets, the last applicable one wins."""
        cfg = _make_cfg(
            population_size_decay=[[1, 200], [3, 150], [5, 100], [7, 50], [9, 25]],
        )
        strategy = self._make_strategy(cfg, MockGenePool)

        strategy.on_generation_start(1)
        self.assertEqual(cfg.hyperparameters.population_size.value, 200)

        strategy.on_generation_start(4)
        self.assertEqual(cfg.hyperparameters.population_size.value, 150)

        strategy.on_generation_start(5)
        self.assertEqual(cfg.hyperparameters.population_size.value, 100)

        strategy.on_generation_start(8)
        self.assertEqual(cfg.hyperparameters.population_size.value, 50)

        strategy.on_generation_start(9)
        self.assertEqual(cfg.hyperparameters.population_size.value, 25)

        strategy.on_generation_start(100)
        self.assertEqual(cfg.hyperparameters.population_size.value, 25)

    def test_decay_does_not_affect_num_generations(self, MockGenePool):
        """_apply_decay_schedules must not mutate num_generations or other
        config keys beyond the three it is supposed to set."""
        cfg = _make_cfg()
        cfg.hyperparameters.num_generations = OmegaConf.create({"value": 20})
        strategy = self._make_strategy(cfg, MockGenePool)

        for gen in range(1, 21):
            strategy.on_generation_start(gen)
            self.assertEqual(cfg.hyperparameters.num_generations.value, 20)


@patch("neural_architecture_search.strategies.ga_strategy.GenePool")
class TestNaming(unittest.TestCase):
    """Test that _generate_unique_names produces names with the correct
    generation suffix."""

    def _make_strategy(self, cfg, MockGenePool):
        MockGenePool.return_value = MagicMock()
        return GAStrategy(cfg)

    def test_names_end_with_generation_number(self, MockGenePool):
        """Every name should end with _{generation}."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)

        for gen in (1, 5, 12, 20):
            strategy.on_generation_start(gen)
            names = strategy._generate_unique_names(10)
            for name in names:
                self.assertTrue(
                    name.endswith(f"_{gen}"),
                    f"Name '{name}' should end with '_{gen}' for generation {gen}",
                )

    def test_names_are_unique(self, MockGenePool):
        """All generated names within a generation must be unique."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(3)
        names = strategy._generate_unique_names(50)
        self.assertEqual(len(names), len(set(names)))

    def test_names_are_sorted(self, MockGenePool):
        """Names should be returned in sorted order."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(1)
        names = strategy._generate_unique_names(20)
        self.assertEqual(names, sorted(names))

    def test_name_format(self, MockGenePool):
        """Names should follow the pattern: adjective_animal_N."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(7)
        names = strategy._generate_unique_names(5)
        for name in names:
            parts = name.rsplit("_", 1)
            self.assertEqual(len(parts), 2,
                             f"Name '{name}' should have format 'word_word_N'")
            self.assertEqual(parts[1], "7")
            # The prefix part should contain at least one underscore
            # (adjective_animal)
            self.assertIn("_", parts[0])

    def test_generation_1_names_dont_have_suffix_2(self, MockGenePool):
        """Regression: generation 1 names must NOT end with _2."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(1)
        names = strategy._generate_unique_names(10)
        for name in names:
            self.assertFalse(
                name.endswith("_2"),
                f"Name '{name}' incorrectly ends with _2 for generation 1",
            )
            self.assertTrue(name.endswith("_1"))

    def test_generation_12_names_dont_have_suffix_13(self, MockGenePool):
        """Regression: generation 12 names must NOT end with _13."""
        cfg = _make_cfg()
        strategy = self._make_strategy(cfg, MockGenePool)
        strategy.on_generation_start(12)
        names = strategy._generate_unique_names(10)
        for name in names:
            self.assertFalse(
                name.endswith("_13"),
                f"Name '{name}' incorrectly ends with _13 for generation 12",
            )
            self.assertTrue(name.endswith("_12"))


@patch("neural_architecture_search.strategies.ga_strategy.GenePool")
class TestAskInitialUsesDecayedPopulationSize(unittest.TestCase):
    """Test that ask() in the first generation uses the correct
    population size from the decay schedule."""

    def test_first_generation_respects_decay(self, MockGenePool):
        """ask() with n=population_size.value should produce 100 candidates
        in generation 1, not 50."""
        mock_pool = MagicMock()
        mock_pool.create_gene_sequence.return_value = [
            {"layer": "C_1D", "f_name": "get_conv1d_block", "filters": 16,
             "kernel_size": 3},
        ]
        MockGenePool.return_value = mock_pool

        cfg = _make_cfg()
        strategy = GAStrategy(cfg)
        strategy.on_generation_start(1)

        n = cfg.hyperparameters.population_size.value
        self.assertEqual(n, 100, "Generation 1 should have population_size=100")

        candidates = strategy.ask(n)
        self.assertEqual(len(candidates), 100)

    def test_generation_10_respects_decay(self, MockGenePool):
        """ask() in generation 10 should produce 25 candidates."""
        mock_pool = MagicMock()
        mock_pool.create_gene_sequence.return_value = [
            {"layer": "C_1D", "f_name": "get_conv1d_block", "filters": 16,
             "kernel_size": 3},
        ]
        MockGenePool.return_value = mock_pool

        cfg = _make_cfg()
        strategy = GAStrategy(cfg)
        strategy.on_generation_start(10)

        n = cfg.hyperparameters.population_size.value
        self.assertEqual(n, 25)

        candidates = strategy.ask(n)
        self.assertEqual(len(candidates), 25)


if __name__ == "__main__":
    unittest.main()
