"""
Surrogate Model for Neural Architecture Search.

Predicts validation accuracy from architecture encodings to pre-screen
individuals and skip training for those confidently predicted to perform poorly.

Supported model types:
- ``random_forest`` — Tree-variance across the ensemble provides uncertainty.
  Robust, fast, good default.
- ``gaussian_process`` — Bayesian posterior gives calibrated uncertainty.
  Best for small datasets but scales as O(n³).
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np

VALID_MODEL_TYPES = ("random_forest", "gaussian_process")


class SurrogateModel:
    """
    Surrogate model that predicts validation accuracy from architecture encodings.

    Supports Random Forest and Gaussian Process backends (scikit-learn).
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        min_samples_to_train: int = 20,
        confidence_threshold: float = 0.5,
        exploration_ratio: float = 0.2,
        evaluation_mode: bool = False,
    ):
        """
        Args:
            model_type: Model backend — ``"random_forest"`` or
                ``"gaussian_process"``.
            n_estimators: Number of trees (Random Forest only).
            min_samples_to_train: Minimum number of observations before the
                surrogate begins making predictions.
            confidence_threshold: Predicted accuracy below this value marks an
                individual as a skip candidate.
            exploration_ratio: Fraction of individuals always trained regardless
                of predictions (prevents self-reinforcing bias).
            evaluation_mode: When True, the surrogate predicts for all
                individuals but never skips any — everyone is still fully
                trained. Produces ground-truth comparison data.
        """
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Must be one of {VALID_MODEL_TYPES}."
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.min_samples_to_train = min_samples_to_train
        self.confidence_threshold = confidence_threshold
        self.exploration_ratio = exploration_ratio
        self.evaluation_mode = evaluation_mode

        self._model = None
        self._encodings: List[np.ndarray] = []
        self._accuracies: List[float] = []
        self._is_fitted = False

        # Store predictions from last prescreen for logging
        self._last_predictions: Dict[str, Dict[str, Any]] = {}

    @property
    def is_ready(self) -> bool:
        """Whether the surrogate has enough data to make predictions."""
        return len(self._encodings) >= self.min_samples_to_train

    @property
    def sample_count(self) -> int:
        """Number of training observations collected."""
        return len(self._encodings)

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    def add_observation(self, encoding: np.ndarray, val_acc: float) -> None:
        """
        Add a single observation (encoding, accuracy) pair.

        Args:
            encoding: Architecture encoding vector.
            val_acc: Validation accuracy achieved by this architecture.
        """
        self._encodings.append(np.asarray(encoding, dtype=np.float32))
        self._accuracies.append(float(val_acc))

    def add_observations_batch(
        self, encodings: np.ndarray, val_accs: List[float]
    ) -> None:
        """
        Add a batch of observations.

        Args:
            encodings: 2D array of shape (n_samples, encoding_dim).
            val_accs: List of validation accuracies.
        """
        encodings = np.asarray(encodings, dtype=np.float32)
        for i in range(len(val_accs)):
            self._encodings.append(encodings[i])
            self._accuracies.append(float(val_accs[i]))

    def fit(self) -> None:
        """
        Train (or retrain) the surrogate on all collected observations.

        Raises:
            RuntimeError: If not enough samples have been collected.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Need at least {self.min_samples_to_train} samples to train, "
                f"have {self.sample_count}."
            )

        X = np.array(self._encodings, dtype=np.float32)
        y = np.array(self._accuracies, dtype=np.float32)

        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            self._model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gaussian_process":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            self._model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                random_state=42,
                normalize_y=True,
            )

        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, encoding: np.ndarray) -> Tuple[float, float]:
        """
        Predict accuracy and uncertainty for a single encoding.

        Args:
            encoding: Architecture encoding vector.

        Returns:
            (predicted_accuracy, uncertainty) — uncertainty is tree-variance
            for Random Forest, posterior std for Gaussian Process.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        encoding = np.asarray(encoding, dtype=np.float32).reshape(1, -1)
        pred, unc = self._predict_internal(encoding)
        return float(pred[0]), float(unc[0])

    def predict_batch(
        self, encodings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict accuracy and uncertainty for a batch of encodings.

        Args:
            encodings: 2D array of shape (n_samples, encoding_dim).

        Returns:
            (predicted_accs, uncertainties) arrays of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        encodings = np.asarray(encodings, dtype=np.float32)
        return self._predict_internal(encodings)

    def _predict_internal(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch prediction to the appropriate backend."""
        if self.model_type == "random_forest":
            tree_predictions = np.array(
                [tree.predict(X) for tree in self._model.estimators_]
            )
            return np.mean(tree_predictions, axis=0), np.std(tree_predictions, axis=0)
        elif self.model_type == "gaussian_process":
            pred, std = self._model.predict(X, return_std=True)
            return np.asarray(pred), np.asarray(std)

    def prescreen(
        self,
        individuals: dict,
        encode_fn: Callable,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Partition individuals into to_train, to_skip, and exploration_set.

        Args:
            individuals: Dict mapping individual names to their data
                (must contain 'genotype' key).
            encode_fn: Function that takes a chromosome and returns an encoding.

        Returns:
            (to_train, to_skip, exploration_set) — lists of individual names.
            exploration_set is a subset of to_train.
        """
        names = list(individuals.keys())

        # Not enough data yet — train everyone
        if not self._is_fitted:
            self._last_predictions = {}
            return names, [], []

        # Encode all individuals
        encodings = np.array(
            [encode_fn(individuals[name]["genotype"]) for name in names],
            dtype=np.float32,
        )

        predicted_accs, uncertainties = self.predict_batch(encodings)
        median_uncertainty = float(np.median(uncertainties))

        # Store predictions for logging
        self._last_predictions = {}
        for i, name in enumerate(names):
            self._last_predictions[name] = {
                "predicted_acc": float(predicted_accs[i]),
                "uncertainty": float(uncertainties[i]),
            }

        # Select exploration set (always trained)
        n_explore = max(1, int(len(names) * self.exploration_ratio))
        explore_indices = np.random.choice(
            len(names), size=min(n_explore, len(names)), replace=False
        )
        exploration_set = [names[i] for i in explore_indices]

        # Evaluation mode: predict for all but skip none
        if self.evaluation_mode:
            return names, [], exploration_set

        # Determine skip candidates
        to_skip = []
        to_train = []
        for i, name in enumerate(names):
            if name in exploration_set:
                to_train.append(name)
            elif (
                predicted_accs[i] < self.confidence_threshold
                and uncertainties[i] < median_uncertainty
            ):
                to_skip.append(name)
            else:
                to_train.append(name)

        return to_train, to_skip, exploration_set

    def get_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get predictions from the last prescreen call."""
        return self._last_predictions

    def save(self, path: str) -> None:
        """
        Save the surrogate model to disk.

        Args:
            path: Directory to save into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save training data
        np.savez(
            path / "training_data.npz",
            encodings=np.array(self._encodings, dtype=np.float32) if self._encodings else np.array([], dtype=np.float32),
            accuracies=np.array(self._accuracies, dtype=np.float32),
        )

        # Save model
        if self._is_fitted:
            joblib.dump(self._model, path / "model.joblib")

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "min_samples_to_train": self.min_samples_to_train,
            "confidence_threshold": self.confidence_threshold,
            "exploration_ratio": self.exploration_ratio,
            "evaluation_mode": self.evaluation_mode,
            "sample_count": self.sample_count,
            "is_fitted": self._is_fitted,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """
        Load a surrogate model from disk.

        Args:
            path: Directory to load from.

        Returns:
            SurrogateModel instance.
        """
        path = Path(path)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        model = cls(
            model_type=metadata.get("model_type", "random_forest"),
            n_estimators=metadata["n_estimators"],
            min_samples_to_train=metadata["min_samples_to_train"],
            confidence_threshold=metadata["confidence_threshold"],
            exploration_ratio=metadata["exploration_ratio"],
            evaluation_mode=metadata["evaluation_mode"],
        )

        # Load training data
        data = np.load(path / "training_data.npz")
        encodings = data["encodings"]
        accuracies = data["accuracies"]
        if len(accuracies) > 0:
            for i in range(len(accuracies)):
                model._encodings.append(encodings[i])
                model._accuracies.append(float(accuracies[i]))

        # Load fitted model
        model_path = path / "model.joblib"
        if model_path.exists() and metadata.get("is_fitted", False):
            model._model = joblib.load(model_path)
            model._is_fitted = True

        return model

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about the surrogate model.

        Returns:
            Dict with sample_count, accuracy_stats, model_type, and
            model-specific info (feature_importances for RF, kernel_params
            for GP).
        """
        diag: Dict[str, Any] = {
            "model_type": self.model_type,
            "sample_count": self.sample_count,
            "is_fitted": self._is_fitted,
            "is_ready": self.is_ready,
        }

        if self._accuracies:
            accs = np.array(self._accuracies)
            diag["accuracy_stats"] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
            }

        if self._is_fitted:
            if self.model_type == "random_forest":
                importances = self._model.feature_importances_
                top_k = min(10, len(importances))
                top_indices = np.argsort(importances)[-top_k:][::-1]
                diag["feature_importances"] = {
                    int(idx): float(importances[idx]) for idx in top_indices
                }
            elif self.model_type == "gaussian_process":
                diag["kernel_params"] = str(self._model.kernel_)

        return diag

    def log_generation(
        self,
        generation: int,
        individual_records: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """
        Log per-individual and per-generation surrogate statistics.

        Appends rows to surrogate_log.csv and updates surrogate_summary.csv.

        Args:
            generation: Current generation number.
            individual_records: List of dicts, each with keys:
                name, predicted_acc, uncertainty, actual_acc, skipped
            path: Directory to write CSV files into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        log_path = path / "surrogate_log.csv"
        summary_path = path / "surrogate_summary.csv"

        # Write per-individual log
        log_exists = log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow([
                    "generation", "individual", "predicted_acc",
                    "uncertainty", "actual_acc", "skipped",
                ])
            for rec in individual_records:
                writer.writerow([
                    generation,
                    rec["name"],
                    f"{rec['predicted_acc']:.6f}" if rec["predicted_acc"] is not None else "",
                    f"{rec['uncertainty']:.6f}" if rec["uncertainty"] is not None else "",
                    f"{rec['actual_acc']:.6f}" if rec["actual_acc"] is not None else "",
                    rec["skipped"],
                ])

        # Compute summary statistics
        n_total = len(individual_records)
        n_skipped = sum(1 for r in individual_records if r["skipped"])
        n_trained = n_total - n_skipped

        # Compute MAE and correlation for records that have both predicted and actual
        paired = [
            (r["predicted_acc"], r["actual_acc"])
            for r in individual_records
            if r["predicted_acc"] is not None and r["actual_acc"] is not None
        ]

        mae = ""
        correlation = ""
        r_squared = ""

        if len(paired) >= 2:
            pred_arr = np.array([p[0] for p in paired])
            actual_arr = np.array([p[1] for p in paired])
            mae = f"{float(np.mean(np.abs(pred_arr - actual_arr))):.6f}"

            if np.std(pred_arr) > 0 and np.std(actual_arr) > 0:
                corr = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
                correlation = f"{corr:.6f}"
                r_squared = f"{corr ** 2:.6f}"

        summary_exists = summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not summary_exists:
                writer.writerow([
                    "generation", "n_total", "n_skipped", "n_trained",
                    "mae", "correlation", "r_squared",
                ])
            writer.writerow([
                generation, n_total, n_skipped, n_trained,
                mae, correlation, r_squared,
            ])

        # Update the evaluation plot
        self.plot_evaluation(str(path))

    @staticmethod
    def plot_evaluation(surrogate_dir: str) -> None:
        """
        Generate a 4-panel evaluation plot from the surrogate CSV logs.

        Overwrites surrogate_evaluation.png each time it is called.

        Args:
            surrogate_dir: Directory containing surrogate_log.csv and
                surrogate_summary.csv.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        surrogate_dir = Path(surrogate_dir)
        log_path = surrogate_dir / "surrogate_log.csv"
        summary_path = surrogate_dir / "surrogate_summary.csv"

        if not log_path.exists() or not summary_path.exists():
            return

        with open(log_path, "r") as f:
            log_rows = list(csv.DictReader(f))
        with open(summary_path, "r") as f:
            summary_rows = list(csv.DictReader(f))

        # Filter rows with both predicted and actual accuracy
        pred_rows = [
            r for r in log_rows
            if r["predicted_acc"] and r["actual_acc"]
        ]
        if not pred_rows:
            return

        pred_acc = np.array([float(r["predicted_acc"]) for r in pred_rows])
        actual_acc = np.array([float(r["actual_acc"]) for r in pred_rows])
        generations = np.array([int(r["generation"]) for r in pred_rows])

        sum_gens = [int(r["generation"]) for r in summary_rows if r["mae"]]
        sum_mae = [float(r["mae"]) for r in summary_rows if r["mae"]]
        sum_corr = [float(r["correlation"]) for r in summary_rows if r["correlation"]]
        sum_r2 = [float(r["r_squared"]) for r in summary_rows if r["r_squared"]]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Surrogate Model Evaluation", fontsize=14, fontweight="bold")

        # 1. Scatter: predicted vs actual
        ax = axes[0, 0]
        unique_gens = sorted(set(generations))
        cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_gens)))
        for gen, color in zip(unique_gens, cmap):
            mask = generations == gen
            ax.scatter(actual_acc[mask], pred_acc[mask],
                       c=[color], alpha=0.6, s=20, label=f"Gen {gen}")
        lims = [0, max(actual_acc.max(), pred_acc.max()) + 0.05]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Actual Accuracy")
        ax.set_ylabel("Predicted Accuracy")
        ax.set_title("Predicted vs Actual Accuracy")
        ax.legend(fontsize=7, loc="upper left")

        # 2. Per-generation correlation & R²
        ax = axes[0, 1]
        if sum_corr:
            ax.plot(sum_gens, sum_corr, "o-", color="tab:blue", label="Correlation")
            ax.plot(sum_gens[:len(sum_r2)], sum_r2, "s--", color="tab:orange", label="R²")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Value")
        ax.set_title("Per-Generation Correlation & R²")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)

        # 3. Per-generation MAE
        ax = axes[1, 0]
        if sum_mae:
            ax.plot(sum_gens, sum_mae, "o-", color="tab:red")
        ax.set_xlabel("Generation")
        ax.set_ylabel("MAE")
        ax.set_title("Per-Generation MAE")

        # 4. Error distribution
        ax = axes[1, 1]
        errors = pred_acc - actual_acc
        ax.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="tab:green")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Prediction Error (predicted − actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"Error Distribution (mean={errors.mean():.3f}, std={errors.std():.3f})")

        plt.tight_layout()
        plt.savefig(surrogate_dir / "surrogate_evaluation.png", dpi=150)
        plt.close()
