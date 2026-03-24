"""Tests for HardwareLUT."""

import json
import os
import tempfile

import numpy as np
import pytest

from neural_architecture_search.src.hardware_lut import HardwareLUT, METRIC_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chromosome(n_layers=5, seed=0):
    """Create a synthetic chromosome with realistic layer types."""
    rng = np.random.RandomState(seed)
    layer_types = [
        ("STFT_2D", {"n_fft": 128, "hop_length": 256}),
        ("C_2D_BLOCK", {"filters": int(rng.randint(4, 64)), "kernel_height": int(rng.randint(1, 12)),
                        "kernel_width": int(rng.randint(1, 12)), "strides": int(rng.choice([1, 2]))}),
        ("DC_2D_BLOCK", {"kernel_height": int(rng.randint(1, 12)), "kernel_width": int(rng.randint(1, 12)),
                         "strides": int(rng.choice([1, 2]))}),
        ("D", {"units": int(rng.randint(8, 64)), "activation": "relu"}),
        ("GAP_2D", {}),
    ]
    chromosome = []
    for i in range(n_layers):
        lt, params = layer_types[i % len(layer_types)]
        gene = {"layer": lt, "f_name": lt, **params}
        chromosome.append(gene)
    return chromosome


class FakeRegistry:
    """Minimal registry stub for testing."""

    def __init__(self, slot_size=34, max_layers=16):
        self.max_layers = max_layers

        class Schema:
            pass

        self.schema = Schema()
        self.schema.slot_size = slot_size
        self.schema.layer_vector_size = 14
        self.schema.param_vector_size = slot_size - 14
        self.schema.total_vector_size = slot_size * max_layers
        self.schema.layer_names = [
            "AP_2D", "C_2D_BLOCK", "D", "DC_2D_BLOCK", "GAP_2D",
            "GMP_2D", "MAG_2D", "MP_2D", "STFT_2D",
        ]

    def encode(self, chromosome):
        """Deterministic encoding based on chromosome contents."""
        rng = np.random.RandomState(len(chromosome))
        vec = np.zeros(self.schema.total_vector_size, dtype=np.float32)
        for i, gene in enumerate(chromosome):
            slot_start = i * self.schema.slot_size
            # Put something non-zero in the slot
            lt = gene.get("layer", "UNKNOWN")
            if lt in self.schema.layer_names:
                idx = self.schema.layer_names.index(lt) + 1
                vec[slot_start + idx] = 1.0
            # Fill param section with deterministic values
            param_start = slot_start + self.schema.layer_vector_size
            for j, (k, v) in enumerate(sorted(gene.items())):
                if k in ("layer", "f_name"):
                    continue
                if isinstance(v, (int, float)):
                    pos = param_start + (j % self.schema.param_vector_size)
                    vec[pos] = float(v) / 100.0
        return vec


def _create_synthetic_results(tmpdir, n_individuals=30, seed=42):
    """Create a fake results directory with Generation_1/individual/results.json."""
    rng = np.random.RandomState(seed)
    gen_dir = os.path.join(tmpdir, "Generation_1")
    os.makedirs(gen_dir)

    board_snr = "12345"
    chromosomes = []

    for i in range(n_individuals):
        name = f"individual_{i}"
        ind_dir = os.path.join(gen_dir, name)
        os.makedirs(ind_dir)

        chromosome = _make_chromosome(n_layers=rng.randint(3, 10), seed=i)
        chromosomes.append(chromosome)

        with open(os.path.join(ind_dir, "chromosome.json"), "w") as f:
            json.dump(chromosome, f)

        # Create realistic hardware metrics
        n_layers = len(chromosome)
        results = {
            "memory_footprint_tflite": int(rng.uniform(50000, 500000)),
            "val_acc": float(rng.uniform(0.3, 0.9)),
            "energy_information": {board_snr: float(rng.uniform(5.0, 50.0))},
            "inference_information": {board_snr: int(rng.uniform(500, 5000))},
            "rom_usage": int(rng.uniform(200000, 800000)),
        }

        with open(os.path.join(ind_dir, "results.json"), "w") as f:
            json.dump(results, f)

    return board_snr, chromosomes


# ---------------------------------------------------------------------------
# Tests: Full mode
# ---------------------------------------------------------------------------

class TestBuildFullMode:
    def test_build_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, chromosomes = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )

            assert lut.mode == "full"
            assert len(lut.models) > 0

            # Predict for a new chromosome
            chrom = _make_chromosome(n_layers=6, seed=999)
            encoding = registry.encode(chrom)
            pred = lut.predict(chrom, encoding)

            for metric in pred:
                value, uncertainty = pred[metric]
                assert isinstance(value, float)
                assert isinstance(uncertainty, float)
                assert value >= 0.0

    def test_predict_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, chromosomes = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )

            chroms = [_make_chromosome(n_layers=5, seed=i) for i in range(10)]
            encodings = np.array([registry.encode(c) for c in chroms])
            batch_pred = lut.predict_batch(chroms, encodings)

            for metric in batch_pred:
                preds, uncs = batch_pred[metric]
                assert preds.shape == (10,)
                assert uncs.shape == (10,)
                assert all(p >= 0.0 for p in preds)


# ---------------------------------------------------------------------------
# Tests: Layerwise mode
# ---------------------------------------------------------------------------

class TestBuildLayerwiseMode:
    def test_build_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, chromosomes = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="layerwise",
            )

            assert lut.mode == "layerwise"
            assert len(lut.models) > 0

            chrom = _make_chromosome(n_layers=6, seed=999)
            encoding = registry.encode(chrom)
            pred = lut.predict(chrom, encoding)

            for metric in pred:
                value, uncertainty = pred[metric]
                assert isinstance(value, float)
                assert isinstance(uncertainty, float)
                assert value >= 0.0

    def test_predict_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, chromosomes = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="layerwise",
            )

            chrom = _make_chromosome(n_layers=6, seed=999)
            encoding = registry.encode(chrom)
            breakdown = lut.predict_breakdown(chrom, encoding)

            assert isinstance(breakdown, dict)
            for metric, per_layer in breakdown.items():
                assert isinstance(per_layer, dict)
                assert len(per_layer) > 0
                for key, cost in per_layer.items():
                    assert isinstance(cost, float)
                    assert cost >= 0.0

    def test_breakdown_sums_close_to_total(self):
        """Per-layer costs should sum approximately to the total prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, chromosomes = _create_synthetic_results(tmpdir, n_individuals=50)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="layerwise",
            )

            chrom = _make_chromosome(n_layers=6, seed=999)
            encoding = registry.encode(chrom)

            pred = lut.predict(chrom, encoding)
            breakdown = lut.predict_breakdown(chrom, encoding)

            for metric in pred:
                if metric in breakdown:
                    total_pred = pred[metric][0]
                    breakdown_sum = sum(breakdown[metric].values())
                    # Should be reasonably close (same model, same data)
                    assert abs(total_pred - breakdown_sum) < 1e-6, (
                        f"{metric}: total={total_pred}, breakdown_sum={breakdown_sum}"
                    )


# ---------------------------------------------------------------------------
# Tests: Save/Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_full_mode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, _ = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )

            save_dir = os.path.join(tmpdir, "lut_saved")
            lut.save(save_dir)

            loaded = HardwareLUT.load(save_dir)
            assert loaded.mode == "full"

            chrom = _make_chromosome(n_layers=5, seed=42)
            encoding = registry.encode(chrom)
            pred_orig = lut.predict(chrom, encoding)
            pred_loaded = loaded.predict(chrom, encoding)

            for metric in pred_orig:
                assert metric in pred_loaded
                np.testing.assert_allclose(
                    pred_orig[metric][0], pred_loaded[metric][0], rtol=1e-5,
                )

    def test_layerwise_mode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, _ = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="layerwise",
            )

            save_dir = os.path.join(tmpdir, "lut_saved")
            lut.save(save_dir)

            loaded = HardwareLUT.load(save_dir)
            assert loaded.mode == "layerwise"

            chrom = _make_chromosome(n_layers=5, seed=42)
            encoding = registry.encode(chrom)
            pred_orig = lut.predict(chrom, encoding)
            pred_loaded = loaded.predict(chrom, encoding)

            for metric in pred_orig:
                assert metric in pred_loaded
                np.testing.assert_allclose(
                    pred_orig[metric][0], pred_loaded[metric][0], rtol=1e-5,
                )


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_metrics_graceful(self):
        """Build LUT when some individuals lack certain metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_dir = os.path.join(tmpdir, "Generation_1")
            os.makedirs(gen_dir)
            board_snr = "12345"
            rng = np.random.RandomState(42)

            for i in range(20):
                ind_dir = os.path.join(gen_dir, f"ind_{i}")
                os.makedirs(ind_dir)
                chrom = _make_chromosome(n_layers=4, seed=i)
                with open(os.path.join(ind_dir, "chromosome.json"), "w") as f:
                    json.dump(chrom, f)

                results = {
                    "memory_footprint_tflite": 100000,
                    "val_acc": float(rng.uniform(0.3, 0.8)),
                }
                # Only half have energy data
                if i < 10:
                    results["energy_information"] = {board_snr: float(rng.uniform(5, 50))}
                # All have inference time
                results["inference_information"] = {board_snr: int(rng.uniform(500, 5000))}
                # None have ROM
                with open(os.path.join(ind_dir, "results.json"), "w") as f:
                    json.dump(results, f)

            registry = FakeRegistry()
            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )

            # energy should have a model (10 samples >= 5)
            assert "energy" in lut.models
            # inference_time should have a model
            assert "inference_time" in lut.models
            # rom should be skipped (0 valid samples)
            assert "rom" not in lut.models

    def test_no_results_raises(self):
        """build_from_results raises when directory has no valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FakeRegistry()
            with pytest.raises(ValueError, match="No valid results"):
                HardwareLUT.build_from_results(tmpdir, "999", registry)

    def test_predict_breakdown_raises_in_full_mode(self):
        """predict_breakdown() should raise in full mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, _ = _create_synthetic_results(tmpdir)
            registry = FakeRegistry()
            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )
            chrom = _make_chromosome(n_layers=5)
            encoding = registry.encode(chrom)
            with pytest.raises(RuntimeError, match="layerwise"):
                lut.predict_breakdown(chrom, encoding)


# ---------------------------------------------------------------------------
# Tests: Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_cv_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            board_snr, _ = _create_synthetic_results(tmpdir, n_individuals=30)
            registry = FakeRegistry()

            lut = HardwareLUT.build_from_results(
                tmpdir, board_snr, registry, mode="full",
            )

            cv = lut.cross_validate(tmpdir, board_snr, registry, n_folds=3)
            assert len(cv) > 0
            for metric, scores in cv.items():
                assert "r2" in scores
                assert "mae" in scores
                assert isinstance(scores["r2"], float)
                assert isinstance(scores["mae"], float)
                assert scores["mae"] >= 0.0
