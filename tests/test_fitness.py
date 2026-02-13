import unittest
from omegaconf import OmegaConf

from neural_architecture_search.src.objective_function import calculate_fitness


class TestCalculateFitness(unittest.TestCase):

    def test_accuracy_only(self):
        """When board model is None, fitness equals val_acc."""
        cfg = OmegaConf.create({
            "boards": {"value": [{"model": None, "snr": 0}]},
        })
        results = {"val_acc": 0.95}
        fitness, error = calculate_fitness(results, cfg)
        self.assertAlmostEqual(fitness, 0.95)
        self.assertFalse(error)

    def test_mcu_evaluation(self):
        """Weighted fitness from accuracy, ROM usage, and energy."""
        cfg = OmegaConf.create({
            "boards": {"value": [{"model": "some_model", "snr": 10}]},
            "hyperparameters": {
                "acc_weight": {"value": 0.5},
                "rom_usage_weight": {"value": 0.3},
                "min_rom_usage": {"value": 100},
                "energy_information_weight": {"value": 0.2},
                "min_energy_information": {"value": 50},
            },
        })
        results = {
            "val_acc": 0.9,
            "rom_usage": 200,
            "energy_information": {10: 100},
        }
        fitness, error = calculate_fitness(results, cfg)

        # acc_weighted = 0.9 * 0.5 = 0.45
        # rom_usage_scaled = min(100/200, 1) = 0.5 -> weighted = 0.5 * 0.3 = 0.15
        # energy_scaled = min(50/100, 1) = 0.5 -> weighted = 0.5 * 0.2 = 0.1
        # total = 0.45 + 0.15 + 0.1 = 0.7
        self.assertAlmostEqual(fitness, 0.7)
        self.assertFalse(error)

    def test_error_handling(self):
        """Missing keys in results should return 0 fitness with error flag."""
        cfg = OmegaConf.create({
            "boards": {"value": [{"model": "some_model", "snr": 10}]},
            "hyperparameters": {
                "acc_weight": {"value": 0.5},
                "rom_usage_weight": {"value": 0.3},
                "min_rom_usage": {"value": 100},
                "energy_information_weight": {"value": 0.2},
                "min_energy_information": {"value": 50},
            },
        })
        results = {}
        fitness, error = calculate_fitness(results, cfg)
        self.assertEqual(fitness, 0)
        self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
