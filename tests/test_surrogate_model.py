"""Tests for the SurrogateModel class."""

import csv
import os
import tempfile
import unittest

import numpy as np

from neural_architecture_search.src.surrogate_model import SurrogateModel, VALID_MODEL_TYPES


class TestSurrogateModelInit(unittest.TestCase):
    """Test initialization and default properties."""

    def test_default_init(self):
        """Test default initialization values."""
        model = SurrogateModel()
        self.assertEqual(model.model_type, "random_forest")
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.min_samples_to_train, 20)
        self.assertEqual(model.confidence_threshold, 0.5)
        self.assertEqual(model.exploration_ratio, 0.2)
        self.assertFalse(model.evaluation_mode)
        self.assertFalse(model.is_fitted)
        self.assertFalse(model.is_ready)
        self.assertEqual(model.sample_count, 0)

    def test_custom_init(self):
        """Test custom initialization values."""
        model = SurrogateModel(
            n_estimators=50,
            min_samples_to_train=10,
            confidence_threshold=0.3,
            exploration_ratio=0.1,
            evaluation_mode=True,
        )
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.min_samples_to_train, 10)
        self.assertEqual(model.confidence_threshold, 0.3)
        self.assertEqual(model.exploration_ratio, 0.1)
        self.assertTrue(model.evaluation_mode)

    def test_invalid_model_type(self):
        """Test that invalid model_type raises ValueError."""
        with self.assertRaises(ValueError):
            SurrogateModel(model_type="xgboost")

    def test_gaussian_process_init(self):
        """Test GP initialization."""
        model = SurrogateModel(model_type="gaussian_process")
        self.assertEqual(model.model_type, "gaussian_process")


class TestDataCollection(unittest.TestCase):
    """Test observation collection."""

    def setUp(self):
        self.model = SurrogateModel(min_samples_to_train=5)

    def test_add_single_observation(self):
        """Test adding a single observation."""
        enc = np.random.rand(10)
        self.model.add_observation(enc, 0.85)
        self.assertEqual(self.model.sample_count, 1)

    def test_add_batch_observations(self):
        """Test adding a batch of observations."""
        encodings = np.random.rand(5, 10)
        accs = [0.7, 0.75, 0.8, 0.85, 0.9]
        self.model.add_observations_batch(encodings, accs)
        self.assertEqual(self.model.sample_count, 5)

    def test_is_ready_threshold(self):
        """Test that is_ready activates at the threshold."""
        for i in range(4):
            self.model.add_observation(np.random.rand(10), 0.5 + i * 0.1)
            self.assertFalse(self.model.is_ready)

        self.model.add_observation(np.random.rand(10), 0.9)
        self.assertTrue(self.model.is_ready)

    def test_mixed_single_and_batch(self):
        """Test mixing single and batch additions."""
        self.model.add_observation(np.random.rand(10), 0.5)
        self.model.add_observations_batch(np.random.rand(3, 10), [0.6, 0.7, 0.8])
        self.assertEqual(self.model.sample_count, 4)


class TestFitAndPredict(unittest.TestCase):
    """Test fitting and prediction."""

    def setUp(self):
        self.model = SurrogateModel(n_estimators=10, min_samples_to_train=5)
        np.random.seed(42)
        # Create synthetic data: accuracy correlates with mean of encoding
        for _ in range(20):
            enc = np.random.rand(10)
            acc = np.clip(np.mean(enc) + np.random.normal(0, 0.05), 0, 1)
            self.model.add_observation(enc, acc)

    def test_fit_succeeds(self):
        """Test that fit succeeds with enough data."""
        self.model.fit()
        self.assertTrue(self.model.is_fitted)

    def test_fit_raises_without_enough_data(self):
        """Test that fit raises RuntimeError without enough data."""
        model = SurrogateModel(min_samples_to_train=100)
        model.add_observation(np.random.rand(10), 0.5)
        with self.assertRaises(RuntimeError):
            model.fit()

    def test_predict_single(self):
        """Test single prediction returns (acc, uncertainty) tuple."""
        self.model.fit()
        pred_acc, uncertainty = self.model.predict(np.random.rand(10))
        self.assertIsInstance(pred_acc, float)
        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(uncertainty, 0.0)

    def test_predict_batch(self):
        """Test batch prediction returns correct shapes."""
        self.model.fit()
        encodings = np.random.rand(5, 10)
        pred_accs, uncertainties = self.model.predict_batch(encodings)
        self.assertEqual(pred_accs.shape, (5,))
        self.assertEqual(uncertainties.shape, (5,))

    def test_predict_raises_if_not_fitted(self):
        """Test that predict raises RuntimeError if model is not fitted."""
        model = SurrogateModel()
        with self.assertRaises(RuntimeError):
            model.predict(np.random.rand(10))

    def test_predict_batch_raises_if_not_fitted(self):
        """Test that predict_batch raises RuntimeError if model is not fitted."""
        model = SurrogateModel()
        with self.assertRaises(RuntimeError):
            model.predict_batch(np.random.rand(3, 10))

    def test_predictions_are_reasonable(self):
        """Test that predictions are within a reasonable range."""
        self.model.fit()
        # High-valued encoding should predict higher accuracy
        high_enc = np.ones(10) * 0.9
        low_enc = np.ones(10) * 0.1
        high_pred, _ = self.model.predict(high_enc)
        low_pred, _ = self.model.predict(low_enc)
        # The model should generally predict higher for higher encodings
        self.assertGreater(high_pred, low_pred)


class TestPrescreen(unittest.TestCase):
    """Test prescreening functionality."""

    def _make_individuals(self, n, dim=10):
        """Helper to create mock individuals."""
        individuals = {}
        for i in range(n):
            individuals[f"ind_{i}"] = {
                "genotype": [{"layer": "CONV", "filters": i}],
            }
        return individuals

    def _encode_fn(self, chromosome):
        """Simple mock encode function."""
        # Just produce a deterministic encoding from the chromosome
        val = chromosome[0].get("filters", 0) / 10.0
        return np.full(10, val, dtype=np.float32)

    def test_unfitted_passthrough(self):
        """Test that unfitted surrogate returns all for training."""
        model = SurrogateModel(min_samples_to_train=100)
        individuals = self._make_individuals(5)
        to_train, to_skip, exploration = model.prescreen(
            individuals, self._encode_fn
        )
        self.assertEqual(len(to_train), 5)
        self.assertEqual(len(to_skip), 0)
        self.assertEqual(len(exploration), 0)

    def test_fitted_prescreen_partition(self):
        """Test that fitted prescreen partitions correctly."""
        np.random.seed(42)
        model = SurrogateModel(
            n_estimators=10,
            min_samples_to_train=5,
            confidence_threshold=0.5,
            exploration_ratio=0.2,
        )

        # Add training data: low-valued encodings have low accuracy
        for i in range(20):
            enc = np.full(10, i / 20.0, dtype=np.float32)
            acc = i / 20.0  # accuracy = encoding value
            model.add_observation(enc, acc)
        model.fit()

        individuals = self._make_individuals(10)
        to_train, to_skip, exploration = model.prescreen(
            individuals, self._encode_fn
        )

        # All individuals should be accounted for
        self.assertEqual(len(to_train) + len(to_skip), 10)

        # Exploration set should be a subset of to_train
        for name in exploration:
            self.assertIn(name, to_train)

    def test_exploration_guarantee(self):
        """Test that at least exploration_ratio fraction is always trained."""
        np.random.seed(42)
        model = SurrogateModel(
            n_estimators=10,
            min_samples_to_train=5,
            confidence_threshold=0.99,  # Very high threshold — try to skip everything
            exploration_ratio=0.5,  # But 50% must be trained
        )

        for i in range(20):
            enc = np.full(10, 0.1, dtype=np.float32)
            model.add_observation(enc, 0.1)
        model.fit()

        individuals = self._make_individuals(10)
        to_train, to_skip, exploration = model.prescreen(
            individuals, self._encode_fn
        )

        # At least 50% should be trained (exploration guarantee)
        self.assertGreaterEqual(len(to_train), 5)
        self.assertGreaterEqual(len(exploration), 5)

    def test_evaluation_mode_skips_none(self):
        """Test that evaluation mode predicts for all but skips none."""
        np.random.seed(42)
        model = SurrogateModel(
            n_estimators=10,
            min_samples_to_train=5,
            confidence_threshold=0.5,
            evaluation_mode=True,
        )

        for i in range(20):
            enc = np.full(10, i / 20.0, dtype=np.float32)
            model.add_observation(enc, i / 20.0)
        model.fit()

        individuals = self._make_individuals(10)
        to_train, to_skip, exploration = model.prescreen(
            individuals, self._encode_fn
        )

        # In evaluation mode: everyone is trained, nobody skipped
        self.assertEqual(len(to_train), 10)
        self.assertEqual(len(to_skip), 0)

        # But predictions should still exist
        predictions = model.get_predictions()
        self.assertEqual(len(predictions), 10)


class TestLogging(unittest.TestCase):
    """Test CSV logging functionality."""

    def test_log_generation_creates_files(self):
        """Test that log_generation creates CSV files."""
        model = SurrogateModel()
        records = [
            {"name": "ind_1", "predicted_acc": 0.8, "uncertainty": 0.05,
             "actual_acc": 0.82, "skipped": False},
            {"name": "ind_2", "predicted_acc": 0.3, "uncertainty": 0.02,
             "actual_acc": None, "skipped": True},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.log_generation(1, records, tmpdir)

            log_path = os.path.join(tmpdir, "surrogate_log.csv")
            summary_path = os.path.join(tmpdir, "surrogate_summary.csv")

            self.assertTrue(os.path.exists(log_path))
            self.assertTrue(os.path.exists(summary_path))

    def test_log_generation_correct_rows(self):
        """Test that surrogate_log.csv has correct rows."""
        model = SurrogateModel()
        records = [
            {"name": "ind_1", "predicted_acc": 0.8, "uncertainty": 0.05,
             "actual_acc": 0.82, "skipped": False},
            {"name": "ind_2", "predicted_acc": 0.3, "uncertainty": 0.02,
             "actual_acc": 0.28, "skipped": True},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.log_generation(1, records, tmpdir)

            log_path = os.path.join(tmpdir, "surrogate_log.csv")
            with open(log_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Header + 2 data rows
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][0], "generation")
            self.assertEqual(rows[1][1], "ind_1")
            self.assertEqual(rows[2][1], "ind_2")

    def test_log_generation_appends(self):
        """Test that multiple log_generation calls append to the same file."""
        model = SurrogateModel()
        records_gen1 = [
            {"name": "ind_1", "predicted_acc": 0.8, "uncertainty": 0.05,
             "actual_acc": 0.82, "skipped": False},
        ]
        records_gen2 = [
            {"name": "ind_2", "predicted_acc": 0.6, "uncertainty": 0.03,
             "actual_acc": 0.58, "skipped": False},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.log_generation(1, records_gen1, tmpdir)
            model.log_generation(2, records_gen2, tmpdir)

            log_path = os.path.join(tmpdir, "surrogate_log.csv")
            with open(log_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Header + 2 data rows
            self.assertEqual(len(rows), 3)

    def test_summary_computes_mae(self):
        """Test that summary CSV computes MAE correctly."""
        model = SurrogateModel()
        records = [
            {"name": "ind_1", "predicted_acc": 0.8, "uncertainty": 0.05,
             "actual_acc": 0.82, "skipped": False},
            {"name": "ind_2", "predicted_acc": 0.6, "uncertainty": 0.03,
             "actual_acc": 0.55, "skipped": False},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.log_generation(1, records, tmpdir)

            summary_path = os.path.join(tmpdir, "surrogate_summary.csv")
            with open(summary_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Header + 1 summary row
            self.assertEqual(len(rows), 2)
            # MAE = mean(|0.8-0.82|, |0.6-0.55|) = mean(0.02, 0.05) = 0.035
            mae = float(rows[1][4])
            self.assertAlmostEqual(mae, 0.035, places=4)

    def test_summary_counts(self):
        """Test that summary correctly counts trained and skipped."""
        model = SurrogateModel()
        records = [
            {"name": "ind_1", "predicted_acc": 0.8, "uncertainty": 0.05,
             "actual_acc": 0.82, "skipped": False},
            {"name": "ind_2", "predicted_acc": 0.3, "uncertainty": 0.02,
             "actual_acc": None, "skipped": True},
            {"name": "ind_3", "predicted_acc": 0.7, "uncertainty": 0.04,
             "actual_acc": 0.68, "skipped": False},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model.log_generation(1, records, tmpdir)

            summary_path = os.path.join(tmpdir, "surrogate_summary.csv")
            with open(summary_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            self.assertEqual(int(rows[1][1]), 3)  # n_total
            self.assertEqual(int(rows[1][2]), 1)  # n_skipped
            self.assertEqual(int(rows[1][3]), 2)  # n_trained


class TestPersistence(unittest.TestCase):
    """Test save/load roundtrip."""

    def test_save_load_unfitted(self):
        """Test saving and loading an unfitted model."""
        model = SurrogateModel(
            n_estimators=50,
            min_samples_to_train=10,
            confidence_threshold=0.4,
            exploration_ratio=0.3,
        )
        model.add_observation(np.array([1.0, 2.0, 3.0]), 0.7)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = SurrogateModel.load(tmpdir)

        self.assertEqual(loaded.model_type, "random_forest")
        self.assertEqual(loaded.n_estimators, 50)
        self.assertEqual(loaded.min_samples_to_train, 10)
        self.assertEqual(loaded.confidence_threshold, 0.4)
        self.assertEqual(loaded.exploration_ratio, 0.3)
        self.assertEqual(loaded.sample_count, 1)
        self.assertFalse(loaded.is_fitted)

    def test_save_load_fitted_predictions_match(self):
        """Test that predictions match after save/load."""
        np.random.seed(42)
        model = SurrogateModel(n_estimators=10, min_samples_to_train=5)

        for _ in range(10):
            enc = np.random.rand(10)
            model.add_observation(enc, np.mean(enc))
        model.fit()

        test_enc = np.random.rand(10)
        pred_before, unc_before = model.predict(test_enc)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = SurrogateModel.load(tmpdir)

        pred_after, unc_after = loaded.predict(test_enc)
        self.assertAlmostEqual(pred_before, pred_after, places=6)
        self.assertAlmostEqual(unc_before, unc_after, places=6)

    def test_save_load_preserves_training_data(self):
        """Test that training data is preserved after save/load."""
        model = SurrogateModel(min_samples_to_train=5)
        for i in range(5):
            model.add_observation(np.random.rand(10), 0.5 + i * 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = SurrogateModel.load(tmpdir)

        self.assertEqual(loaded.sample_count, 5)
        self.assertTrue(loaded.is_ready)


class TestDiagnostics(unittest.TestCase):
    """Test diagnostics output."""

    def test_diagnostics_unfitted(self):
        """Test diagnostics for an unfitted model."""
        model = SurrogateModel()
        diag = model.get_diagnostics()
        self.assertEqual(diag["model_type"], "random_forest")
        self.assertEqual(diag["sample_count"], 0)
        self.assertFalse(diag["is_fitted"])
        self.assertFalse(diag["is_ready"])
        self.assertNotIn("accuracy_stats", diag)
        self.assertNotIn("feature_importances", diag)

    def test_diagnostics_with_data(self):
        """Test diagnostics after adding data."""
        model = SurrogateModel(min_samples_to_train=3)
        for i in range(5):
            model.add_observation(np.random.rand(10), 0.5 + i * 0.1)

        diag = model.get_diagnostics()
        self.assertEqual(diag["sample_count"], 5)
        self.assertIn("accuracy_stats", diag)
        self.assertIn("mean", diag["accuracy_stats"])
        self.assertIn("std", diag["accuracy_stats"])
        self.assertIn("min", diag["accuracy_stats"])
        self.assertIn("max", diag["accuracy_stats"])

    def test_diagnostics_fitted(self):
        """Test diagnostics after fitting."""
        np.random.seed(42)
        model = SurrogateModel(n_estimators=10, min_samples_to_train=5)
        for i in range(10):
            model.add_observation(np.random.rand(10), 0.5 + i * 0.05)
        model.fit()

        diag = model.get_diagnostics()
        self.assertTrue(diag["is_fitted"])
        self.assertIn("feature_importances", diag)
        self.assertGreater(len(diag["feature_importances"]), 0)


class TestGaussianProcessBackend(unittest.TestCase):
    """Test Gaussian Process model backend."""

    def setUp(self):
        np.random.seed(42)
        self.model = SurrogateModel(
            model_type="gaussian_process", min_samples_to_train=5
        )
        for _ in range(20):
            enc = np.random.rand(10)
            acc = np.clip(np.mean(enc) + np.random.normal(0, 0.05), 0, 1)
            self.model.add_observation(enc, acc)

    def test_fit_and_predict(self):
        """Test GP fit and predict."""
        self.model.fit()
        self.assertTrue(self.model.is_fitted)
        pred, unc = self.model.predict(np.random.rand(10))
        self.assertIsInstance(pred, float)
        self.assertIsInstance(unc, float)
        self.assertGreaterEqual(unc, 0.0)

    def test_predict_batch(self):
        """Test GP batch prediction shapes."""
        self.model.fit()
        preds, uncs = self.model.predict_batch(np.random.rand(5, 10))
        self.assertEqual(preds.shape, (5,))
        self.assertEqual(uncs.shape, (5,))

    def test_diagnostics_has_kernel_params(self):
        """Test GP diagnostics include kernel_params, not feature_importances."""
        self.model.fit()
        diag = self.model.get_diagnostics()
        self.assertEqual(diag["model_type"], "gaussian_process")
        self.assertIn("kernel_params", diag)
        self.assertNotIn("feature_importances", diag)

    def test_save_load_roundtrip(self):
        """Test GP save/load preserves model_type and predictions."""
        self.model.fit()
        test_enc = np.random.rand(10)
        pred_before, unc_before = self.model.predict(test_enc)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save(tmpdir)
            loaded = SurrogateModel.load(tmpdir)

        self.assertEqual(loaded.model_type, "gaussian_process")
        pred_after, unc_after = loaded.predict(test_enc)
        self.assertAlmostEqual(pred_before, pred_after, places=5)
        self.assertAlmostEqual(unc_before, unc_after, places=5)

    def test_prescreen_works(self):
        """Test GP prescreen partitions correctly."""
        self.model.fit()
        individuals = {}
        for i in range(10):
            individuals[f"ind_{i}"] = {
                "genotype": [{"layer": "CONV", "filters": i}],
            }

        def encode_fn(chrom):
            val = chrom[0].get("filters", 0) / 10.0
            return np.full(10, val, dtype=np.float32)

        to_train, to_skip, exploration = self.model.prescreen(
            individuals, encode_fn
        )
        self.assertEqual(len(to_train) + len(to_skip), 10)


if __name__ == "__main__":
    unittest.main()
