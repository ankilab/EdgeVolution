"""
Hardware Lookup Table (LUT) for pre-built hardware cost models.

Built from a profiling phase that evaluates random architectures on real
MCU hardware.  Predicts energy, inference time, and ROM for any
architecture — no hardware needed during NAS.

Supports two prediction backends:
- ``full``: One SurrogateModel per metric over the full architecture encoding.
- ``layerwise``: Per-layer-type regressors; total cost = sum of per-layer costs.

Usage::

    # Build from profiling results
    lut = HardwareLUT.build_from_results(results_dir, board_snr, registry)

    # Predict
    pred = lut.predict(chromosome, encoding)
    # => {'energy': (value, uncertainty), 'inference_time': ..., 'rom': ...}

    # Save / load
    lut.save("hardware_luts/nrf52840dk/")
    lut = HardwareLUT.load("hardware_luts/nrf52840dk/")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from .surrogate_model import SurrogateModel

logger = logging.getLogger(__name__)

# Metrics we model
METRIC_NAMES = ("energy", "inference_time", "rom")


class HardwareLUT:
    """Pre-built hardware cost model for a specific MCU target."""

    def __init__(
        self,
        mode: str,
        models: Dict[str, Any],
        registry_info: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        """
        Args:
            mode: ``"full"`` or ``"layerwise"``.
            models: Backend-specific model data.
                - full: ``{metric_name: SurrogateModel}``
                - layerwise: ``{metric_name: {layer_type: sklearn_model}}``
            registry_info: Encoding schema info (slot_size, layer_names, etc.)
            metadata: Board info, sample count, creation date, etc.
        """
        if mode not in ("full", "layerwise"):
            raise ValueError(f"Unknown mode '{mode}'. Must be 'full' or 'layerwise'.")
        self.mode = mode
        self.models = models
        self.registry_info = registry_info
        self.metadata = metadata

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        chromosome: List[Dict[str, Any]],
        encoding: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Predict hardware metrics for an architecture.

        Args:
            chromosome: Genotype (list of gene dicts).
            encoding: Pre-computed encoding vector (optional, used by full mode).

        Returns:
            Dict mapping metric name to ``(prediction, uncertainty)``.
        """
        if self.mode == "full":
            return self._predict_full(encoding)
        else:
            return self._predict_layerwise(chromosome, encoding)

    def predict_batch(
        self,
        chromosomes: List[List[Dict[str, Any]]],
        encodings: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Predict hardware metrics for a batch of architectures.

        Returns:
            Dict mapping metric name to ``(predictions, uncertainties)`` arrays.
        """
        if self.mode == "full":
            return self._predict_batch_full(encodings)
        else:
            return self._predict_batch_layerwise(chromosomes, encodings)

    def predict_breakdown(
        self,
        chromosome: List[Dict[str, Any]],
        encoding: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Per-layer cost breakdown (layerwise mode only).

        Returns:
            Dict mapping metric name to dict of
            ``{layer_idx_type: predicted_cost}``.
        """
        if self.mode != "layerwise":
            raise RuntimeError("predict_breakdown() is only available in layerwise mode.")
        return self._layer_breakdown(chromosome, encoding)

    # ------------------------------------------------------------------
    # Full-architecture backend
    # ------------------------------------------------------------------

    def _predict_full(self, encoding: np.ndarray) -> Dict[str, Tuple[float, float]]:
        result = {}
        for metric, model in self.models.items():
            pred, unc = model.predict(encoding)
            result[metric] = (max(0.0, pred), unc)
        return result

    def _predict_batch_full(
        self, encodings: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        result = {}
        for metric, model in self.models.items():
            preds, uncs = model.predict_batch(encodings)
            result[metric] = (np.maximum(0.0, preds), uncs)
        return result

    # ------------------------------------------------------------------
    # Layer-wise backend
    # ------------------------------------------------------------------

    def _extract_layer_features(
        self, chromosome: List[Dict[str, Any]], encoding: Optional[np.ndarray],
    ) -> List[Tuple[str, np.ndarray]]:
        """Extract per-layer features for the layerwise backend.

        Returns list of (layer_type, feature_vector) tuples.
        Only includes layers within max_layers (those with encoded slots).
        """
        slot_size = self.registry_info["slot_size"]
        max_layers = self.registry_info.get("max_layers", 16)
        n_layers = len(chromosome)

        features_list = []
        for i, gene in enumerate(chromosome):
            if i >= max_layers:
                break  # No encoded representation beyond max_layers

            layer_type = gene.get("layer", "UNKNOWN")

            if encoding is not None:
                slot_start = i * slot_size
                slot_features = encoding[slot_start: slot_start + slot_size].copy()
            else:
                slot_features = np.zeros(slot_size, dtype=np.float32)

            # Append context features: position and total depth
            context = np.array([
                i / max(n_layers - 1, 1),   # normalized position
                n_layers / max_layers,       # normalized depth
            ], dtype=np.float32)

            feature_vec = np.concatenate([slot_features, context])
            features_list.append((layer_type, feature_vec))

        return features_list

    def _predict_layerwise(
        self,
        chromosome: List[Dict[str, Any]],
        encoding: Optional[np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        layer_features = self._extract_layer_features(chromosome, encoding)
        result = {}

        for metric, layer_models in self.models.items():
            total_pred = 0.0
            # Uncertainty: sum of variances → sqrt for combined std
            total_var = 0.0

            for layer_type, features in layer_features:
                model_info = layer_models.get(layer_type)
                if model_info is None:
                    # Unknown layer type — use fallback model if available
                    model_info = layer_models.get("_fallback")
                if model_info is None:
                    continue

                model = model_info["model"]
                scaler = model_info.get("scale_factor", 1.0)
                features_2d = features.reshape(1, -1)
                pred = model.predict(features_2d)[0] * scaler

                # Tree-variance for uncertainty
                if hasattr(model, "estimators_"):
                    tree_preds = np.array([
                        t.predict(features_2d)[0] for t in model.estimators_
                    ]) * scaler
                    total_var += np.var(tree_preds)

                total_pred += pred

            total_pred = max(0.0, total_pred)
            result[metric] = (total_pred, np.sqrt(total_var))

        return result

    def _predict_batch_layerwise(
        self,
        chromosomes: List[List[Dict[str, Any]]],
        encodings: Optional[np.ndarray],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        all_preds = {m: [] for m in self.models}
        all_uncs = {m: [] for m in self.models}

        for i, chrom in enumerate(chromosomes):
            enc = encodings[i] if encodings is not None else None
            pred = self._predict_layerwise(chrom, enc)
            for metric in self.models:
                if metric in pred:
                    all_preds[metric].append(pred[metric][0])
                    all_uncs[metric].append(pred[metric][1])
                else:
                    all_preds[metric].append(0.0)
                    all_uncs[metric].append(0.0)

        return {
            m: (np.array(all_preds[m]), np.array(all_uncs[m]))
            for m in self.models
        }

    def _layer_breakdown(
        self,
        chromosome: List[Dict[str, Any]],
        encoding: Optional[np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        layer_features = self._extract_layer_features(chromosome, encoding)
        breakdown = {}

        for metric, layer_models in self.models.items():
            metric_breakdown = {}
            for i, (layer_type, features) in enumerate(layer_features):
                model_info = layer_models.get(layer_type)
                if model_info is None:
                    model_info = layer_models.get("_fallback")
                if model_info is None:
                    continue

                model = model_info["model"]
                scaler = model_info.get("scale_factor", 1.0)
                pred = model.predict(features.reshape(1, -1))[0] * scaler
                key = f"{i}_{layer_type}"
                metric_breakdown[key] = float(max(0.0, pred))
            breakdown[metric] = metric_breakdown

        return breakdown

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the LUT to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "lut_metadata.json", "w") as f:
            json.dump({**self.metadata, "mode": self.mode}, f, indent=2)

        # Save registry info
        with open(path / "registry_info.json", "w") as f:
            json.dump(self.registry_info, f, indent=2)

        if self.mode == "full":
            for metric, model in self.models.items():
                model.save(str(path / metric))
        else:
            for metric, layer_models in self.models.items():
                metric_dir = path / metric
                metric_dir.mkdir(parents=True, exist_ok=True)
                layer_info = {}
                for layer_type, model_info in layer_models.items():
                    model_path = metric_dir / f"{layer_type}.joblib"
                    joblib.dump(model_info["model"], model_path)
                    layer_info[layer_type] = {
                        "scale_factor": model_info.get("scale_factor", 1.0),
                    }
                with open(metric_dir / "layer_models.json", "w") as f:
                    json.dump(layer_info, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HardwareLUT":
        """Load a LUT from a directory."""
        path = Path(path)

        with open(path / "lut_metadata.json") as f:
            metadata = json.load(f)
        mode = metadata.pop("mode")

        with open(path / "registry_info.json") as f:
            registry_info = json.load(f)

        if mode == "full":
            models = {}
            for metric in METRIC_NAMES:
                metric_dir = path / metric
                if metric_dir.is_dir() and (metric_dir / "metadata.json").exists():
                    models[metric] = SurrogateModel.load(str(metric_dir))
        else:
            models = {}
            for metric in METRIC_NAMES:
                metric_dir = path / metric
                if not metric_dir.is_dir():
                    continue
                layer_info_path = metric_dir / "layer_models.json"
                if not layer_info_path.exists():
                    continue
                with open(layer_info_path) as f:
                    layer_info = json.load(f)
                layer_models = {}
                for layer_type, info in layer_info.items():
                    model_path = metric_dir / f"{layer_type}.joblib"
                    if model_path.exists():
                        layer_models[layer_type] = {
                            "model": joblib.load(model_path),
                            "scale_factor": info.get("scale_factor", 1.0),
                        }
                models[metric] = layer_models

        return cls(mode=mode, models=models, registry_info=registry_info, metadata=metadata)

    # ------------------------------------------------------------------
    # Build from profiling results
    # ------------------------------------------------------------------

    @classmethod
    def build_from_results(
        cls,
        results_dir: str,
        board_snr: str,
        registry,
        mode: str = "full",
        n_estimators: int = 100,
    ) -> "HardwareLUT":
        """Build a HardwareLUT from profiling results on disk.

        Scans ``Generation_*/*/results.json`` and ``chromosome.json`` in
        *results_dir*, extracts hardware metrics, and trains models.

        Args:
            results_dir: Path to the profiling results directory.
            board_snr: Board serial number to extract metrics for.
            registry: SearchSpaceRegistry instance for encoding.
            mode: ``"full"`` or ``"layerwise"``.
            n_estimators: Number of trees for Random Forest models.

        Returns:
            A trained HardwareLUT.
        """
        results_dir = Path(results_dir)

        # Collect data from all generations
        chromosomes = []
        encodings = []
        metrics = {m: [] for m in METRIC_NAMES}
        valid_mask = {m: [] for m in METRIC_NAMES}

        for gen_dir in sorted(results_dir.glob("Generation_*")):
            if not gen_dir.is_dir():
                continue
            for ind_dir in sorted(gen_dir.iterdir()):
                if not ind_dir.is_dir():
                    continue
                chromosome_path = ind_dir / "chromosome.json"
                results_path = ind_dir / "results.json"
                if not chromosome_path.exists() or not results_path.exists():
                    continue

                with open(chromosome_path) as f:
                    chromosome = json.load(f)
                with open(results_path) as f:
                    results = json.load(f)

                encoding = registry.encode(chromosome)
                chromosomes.append(chromosome)
                encodings.append(encoding)

                # Extract energy
                energy_info = results.get("energy_information")
                if isinstance(energy_info, dict):
                    val = energy_info.get(board_snr)
                    if isinstance(val, (int, float)):
                        metrics["energy"].append(float(val))
                        valid_mask["energy"].append(True)
                    else:
                        metrics["energy"].append(0.0)
                        valid_mask["energy"].append(False)
                else:
                    metrics["energy"].append(0.0)
                    valid_mask["energy"].append(False)

                # Extract inference time
                inf_info = results.get("inference_information")
                if isinstance(inf_info, dict):
                    val = inf_info.get(board_snr)
                    if isinstance(val, (int, float)):
                        metrics["inference_time"].append(float(val))
                        valid_mask["inference_time"].append(True)
                    else:
                        metrics["inference_time"].append(0.0)
                        valid_mask["inference_time"].append(False)
                else:
                    metrics["inference_time"].append(0.0)
                    valid_mask["inference_time"].append(False)

                # Extract ROM
                rom = results.get("rom_usage")
                if isinstance(rom, (int, float)):
                    metrics["rom"].append(float(rom))
                    valid_mask["rom"].append(True)
                else:
                    metrics["rom"].append(0.0)
                    valid_mask["rom"].append(False)

        if not encodings:
            raise ValueError(f"No valid results found in {results_dir}")

        encodings_arr = np.array(encodings, dtype=np.float32)

        registry_info = {
            "slot_size": registry.schema.slot_size,
            "layer_vector_size": registry.schema.layer_vector_size,
            "param_vector_size": registry.schema.param_vector_size,
            "max_layers": registry.max_layers,
            "layer_names": registry.schema.layer_names,
            "total_vector_size": registry.schema.total_vector_size,
        }

        metadata = {
            "board_snr": board_snr,
            "source_dir": str(results_dir),
            "total_samples": len(encodings),
        }

        if mode == "full":
            models = cls._build_full_models(
                encodings_arr, metrics, valid_mask, n_estimators,
            )
        else:
            models = cls._build_layerwise_models(
                chromosomes, encodings_arr, metrics, valid_mask,
                registry_info, n_estimators,
            )

        # Add per-metric sample counts and R² to metadata
        for metric in METRIC_NAMES:
            mask = valid_mask[metric]
            metadata[f"{metric}_samples"] = sum(mask)

        return cls(mode=mode, models=models, registry_info=registry_info, metadata=metadata)

    @classmethod
    def _build_full_models(
        cls,
        encodings: np.ndarray,
        metrics: Dict[str, List[float]],
        valid_mask: Dict[str, List[bool]],
        n_estimators: int,
    ) -> Dict[str, SurrogateModel]:
        """Train one SurrogateModel per metric."""
        models = {}
        for metric in METRIC_NAMES:
            mask = np.array(valid_mask[metric])
            if mask.sum() < 5:
                logger.info(f"Skipping metric '{metric}': only {mask.sum()} valid samples.")
                continue

            X = encodings[mask]
            y = np.array(metrics[metric])[mask]

            surrogate = SurrogateModel(
                model_type="random_forest",
                n_estimators=n_estimators,
                min_samples_to_train=min(5, len(y)),
                target_name=metric,
            )
            surrogate.add_observations_batch(X, y.tolist())
            surrogate.fit()
            models[metric] = surrogate
            logger.info(f"  {metric}: trained on {len(y)} samples")

        return models

    @classmethod
    def _build_layerwise_models(
        cls,
        chromosomes: List[List[Dict[str, Any]]],
        encodings: np.ndarray,
        metrics: Dict[str, List[float]],
        valid_mask: Dict[str, List[bool]],
        registry_info: Dict[str, Any],
        n_estimators: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Train per-layer-type regressors for each metric.

        Two-stage approach:
        1. Train per-layer-type regressors with initial target = total / n_layers.
        2. Calibrate scale factors so sum(per-layer) minimizes MSE vs total.
        """
        from sklearn.ensemble import RandomForestRegressor

        slot_size = registry_info["slot_size"]
        max_layers = registry_info.get("max_layers", 16)

        models = {}
        for metric in METRIC_NAMES:
            mask = np.array(valid_mask[metric])
            if mask.sum() < 5:
                logger.info(f"Skipping metric '{metric}': only {mask.sum()} valid samples.")
                continue

            valid_indices = np.where(mask)[0]
            y_total = np.array(metrics[metric])[mask]

            # Collect per-layer features grouped by layer type
            layer_type_features = {}  # layer_type -> list of feature vectors
            layer_type_targets = {}   # layer_type -> list of initial targets (total/n_layers)
            # Also track which architecture each layer belongs to (for calibration)
            layer_type_arch_idx = {}  # layer_type -> list of architecture indices

            for arch_i, global_i in enumerate(valid_indices):
                chrom = chromosomes[global_i]
                enc = encodings[global_i]
                n_layers = len(chrom)
                # Only count encoded layers for target distribution
                n_encoded = min(n_layers, max_layers)
                initial_target = y_total[arch_i] / max(n_encoded, 1)

                for layer_i, gene in enumerate(chrom):
                    if layer_i >= max_layers:
                        break  # No encoded representation beyond max_layers
                    layer_type = gene.get("layer", "UNKNOWN")
                    slot_start = layer_i * slot_size
                    slot_features = enc[slot_start: slot_start + slot_size].copy()
                    context = np.array([
                        layer_i / max(n_layers - 1, 1),
                        n_layers / max_layers,
                    ], dtype=np.float32)
                    feature_vec = np.concatenate([slot_features, context])

                    if layer_type not in layer_type_features:
                        layer_type_features[layer_type] = []
                        layer_type_targets[layer_type] = []
                        layer_type_arch_idx[layer_type] = []

                    layer_type_features[layer_type].append(feature_vec)
                    layer_type_targets[layer_type].append(initial_target)
                    layer_type_arch_idx[layer_type].append(arch_i)

            # Stage 1: Train per-layer-type regressors
            layer_models = {}
            for layer_type in layer_type_features:
                X_lt = np.array(layer_type_features[layer_type], dtype=np.float32)
                y_lt = np.array(layer_type_targets[layer_type], dtype=np.float32)

                if len(X_lt) < 3:
                    # Too few samples — use a simple mean predictor
                    from sklearn.dummy import DummyRegressor
                    model = DummyRegressor(strategy="mean")
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=-1,
                    )
                model.fit(X_lt, y_lt)
                layer_models[layer_type] = {
                    "model": model,
                    "scale_factor": 1.0,
                    "features": X_lt,
                    "arch_indices": layer_type_arch_idx[layer_type],
                }

            # Stage 2: Calibrate scale factors
            # For each architecture, compute sum of per-layer predictions
            n_archs = len(y_total)
            pred_per_arch = np.zeros(n_archs, dtype=np.float64)
            # Track which layer types contribute to each architecture
            layer_type_pred_per_arch = {lt: np.zeros(n_archs) for lt in layer_models}

            for layer_type, info in layer_models.items():
                model = info["model"]
                X_lt = info["features"]
                arch_indices = info["arch_indices"]
                preds = model.predict(X_lt)
                for pred_val, arch_i in zip(preds, arch_indices):
                    pred_per_arch[arch_i] += pred_val
                    layer_type_pred_per_arch[layer_type][arch_i] += pred_val

            # Global scale factor: scale = sum(actual) / sum(predicted)
            total_pred_sum = pred_per_arch.sum()
            if total_pred_sum > 0:
                global_scale = y_total.sum() / total_pred_sum
            else:
                global_scale = 1.0

            # Apply global scale to all layer types
            for layer_type in layer_models:
                layer_models[layer_type]["scale_factor"] = float(global_scale)

            # Clean up temporary data before saving
            for layer_type in layer_models:
                layer_models[layer_type].pop("features", None)
                layer_models[layer_type].pop("arch_indices", None)

            models[metric] = layer_models
            logger.info(f"  {metric}: trained {len(layer_models)} layer-type models "
                        f"on {len(y_total)} samples (scale={global_scale:.3f})")

        return models

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        results_dir: str,
        board_snr: str,
        registry,
        n_folds: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Run k-fold cross-validation and return R²/MAE per metric.

        Args:
            results_dir: Path to profiling results.
            board_snr: Board serial number.
            registry: SearchSpaceRegistry instance.
            n_folds: Number of CV folds.

        Returns:
            Dict mapping metric name to ``{"r2": ..., "mae": ...}``.
        """
        from sklearn.model_selection import KFold

        results_dir = Path(results_dir)

        # Collect all data
        chromosomes = []
        encodings = []
        metric_values = {m: [] for m in METRIC_NAMES}
        mask_values = {m: [] for m in METRIC_NAMES}

        for gen_dir in sorted(results_dir.glob("Generation_*")):
            if not gen_dir.is_dir():
                continue
            for ind_dir in sorted(gen_dir.iterdir()):
                if not ind_dir.is_dir():
                    continue
                chrom_path = ind_dir / "chromosome.json"
                res_path = ind_dir / "results.json"
                if not chrom_path.exists() or not res_path.exists():
                    continue

                with open(chrom_path) as f:
                    chrom = json.load(f)
                with open(res_path) as f:
                    res = json.load(f)

                chromosomes.append(chrom)
                encodings.append(registry.encode(chrom))

                energy_info = res.get("energy_information")
                if isinstance(energy_info, dict):
                    val = energy_info.get(board_snr)
                    has = isinstance(val, (int, float))
                    metric_values["energy"].append(float(val) if has else 0.0)
                    mask_values["energy"].append(has)
                else:
                    metric_values["energy"].append(0.0)
                    mask_values["energy"].append(False)

                inf_info = res.get("inference_information")
                if isinstance(inf_info, dict):
                    val = inf_info.get(board_snr)
                    has = isinstance(val, (int, float))
                    metric_values["inference_time"].append(float(val) if has else 0.0)
                    mask_values["inference_time"].append(has)
                else:
                    metric_values["inference_time"].append(0.0)
                    mask_values["inference_time"].append(False)

                rom = res.get("rom_usage")
                has_rom = isinstance(rom, (int, float))
                metric_values["rom"].append(float(rom) if has_rom else 0.0)
                mask_values["rom"].append(has_rom)

        encodings_arr = np.array(encodings, dtype=np.float32)
        cv_results = {}

        for metric in METRIC_NAMES:
            mask = np.array(mask_values[metric])
            if mask.sum() < n_folds + 1:
                continue

            valid_idx = np.where(mask)[0]
            y = np.array(metric_values[metric])[valid_idx]
            X = encodings_arr[valid_idx]
            valid_chroms = [chromosomes[i] for i in valid_idx]

            kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)
            all_actual = []
            all_pred = []

            for train_idx, test_idx in kf.split(X):
                # Build a temporary LUT on the train fold
                fold_lut = self._build_fold_model(
                    [valid_chroms[i] for i in train_idx],
                    X[train_idx],
                    y[train_idx],
                    metric,
                    registry,
                )

                # Predict on test fold
                for i in test_idx:
                    pred = fold_lut.predict(valid_chroms[i], X[i])
                    if metric in pred:
                        all_pred.append(pred[metric][0])
                        all_actual.append(y[i])

            all_actual = np.array(all_actual)
            all_pred = np.array(all_pred)

            if len(all_actual) > 1:
                mae = float(np.mean(np.abs(all_actual - all_pred)))
                ss_res = np.sum((all_actual - all_pred) ** 2)
                ss_tot = np.sum((all_actual - np.mean(all_actual)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                cv_results[metric] = {"r2": r2, "mae": mae}

        return cv_results

    def _build_fold_model(
        self,
        chromosomes: List,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
        registry,
    ) -> "HardwareLUT":
        """Build a single-metric LUT for one CV fold."""
        if self.mode == "full":
            surrogate = SurrogateModel(
                model_type="random_forest",
                n_estimators=100,
                min_samples_to_train=min(3, len(y)),
                target_name=metric,
            )
            surrogate.add_observations_batch(X, y.tolist())
            surrogate.fit()
            models = {metric: surrogate}
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.dummy import DummyRegressor

            slot_size = self.registry_info["slot_size"]
            max_layers = self.registry_info.get("max_layers", 16)

            layer_type_features = {}
            layer_type_targets = {}
            layer_type_arch_idx = {}

            for arch_i, (chrom, enc) in enumerate(zip(chromosomes, X)):
                n_layers = len(chrom)
                n_encoded = min(n_layers, max_layers)
                initial_target = y[arch_i] / max(n_encoded, 1)
                for layer_i, gene in enumerate(chrom):
                    if layer_i >= max_layers:
                        break
                    layer_type = gene.get("layer", "UNKNOWN")
                    slot_start = layer_i * slot_size
                    slot_features = enc[slot_start: slot_start + slot_size].copy()
                    context = np.array([
                        layer_i / max(n_layers - 1, 1),
                        n_layers / max_layers,
                    ], dtype=np.float32)
                    feature_vector = np.concatenate([slot_features, context])

                    if layer_type not in layer_type_features:
                        layer_type_features[layer_type] = []
                        layer_type_targets[layer_type] = []
                        layer_type_arch_idx[layer_type] = []

                    layer_type_features[layer_type].append(feature_vector)
                    layer_type_targets[layer_type].append(initial_target)
                    layer_type_arch_idx[layer_type].append(arch_i)

            layer_models = {}
            for layer_type in layer_type_features:
                X_lt = np.array(layer_type_features[layer_type], dtype=np.float32)
                y_lt = np.array(layer_type_targets[layer_type], dtype=np.float32)
                if len(X_lt) < 3:
                    model = DummyRegressor(strategy="mean")
                else:
                    model = RandomForestRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1,
                    )
                model.fit(X_lt, y_lt)
                layer_models[layer_type] = {
                    "model": model, "scale_factor": 1.0,
                    "features": X_lt, "arch_indices": layer_type_arch_idx[layer_type],
                }

            # Calibrate
            n_archs = len(y)
            pred_per_arch = np.zeros(n_archs, dtype=np.float64)
            for layer_type, info in layer_models.items():
                preds = info["model"].predict(info["features"])
                for pv, ai in zip(preds, info["arch_indices"]):
                    pred_per_arch[ai] += pv
            total_pred_sum = pred_per_arch.sum()
            global_scale = y.sum() / total_pred_sum if total_pred_sum > 0 else 1.0
            for layer_type in layer_models:
                layer_models[layer_type]["scale_factor"] = float(global_scale)
                layer_models[layer_type].pop("features", None)
                layer_models[layer_type].pop("arch_indices", None)

            models = {metric: layer_models}

        return HardwareLUT(
            mode=self.mode, models=models,
            registry_info=self.registry_info, metadata=self.metadata,
        )
