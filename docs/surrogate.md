# Surrogate Model Configuration

The surrogate model predicts validation accuracy from architecture encodings and optionally skips training for individuals confidently predicted to perform poorly. All surrogate parameters live in the `surrogate` section of `config.yaml` and are disabled by default.

## TL;DR

- Set `surrogate.enabled.value=true` on the command line to turn on surrogate pre-screening.
- Choose a model backend with `surrogate.model_type.value=random_forest` (default) or `surrogate.model_type.value=gaussian_process`.
- Use `surrogate.evaluation_mode.value=true` to predict without skipping (for benchmarking the surrogate itself).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable surrogate-assisted pre-screening. |
| `model_type` | `random_forest` | Model backend (see table below). |
| `n_estimators` | `100` | Number of trees in the Random Forest ensemble. Ignored for Gaussian Process. |
| `min_samples_to_train` | `20` | Minimum observations before the surrogate starts predicting. Until this threshold is reached, all individuals are trained normally. |
| `confidence_threshold` | `0.5` | Predicted accuracy below this value marks an individual as a skip candidate. Lower = more conservative. |
| `exploration_ratio` | `0.2` | Fraction of the population always trained regardless of predictions, preventing self-reinforcing bias. |
| `evaluation_mode` | `false` | When `true`, the surrogate predicts for every individual but never skips any. All individuals are still fully trained. Useful for measuring surrogate accuracy before relying on it. |

## Model Backends

| Backend | `model_type` value | Uncertainty source | Strengths | Limitations |
|---------|-------------------|-------------------|-----------|-------------|
| Random Forest | `random_forest` | Tree-variance across the ensemble | Fast to train, robust to outliers, scales well with data size. Good default. | Uncertainty is not calibrated (not a true posterior). |
| Gaussian Process | `gaussian_process` | Bayesian posterior standard deviation | Calibrated uncertainty estimates, excellent for small datasets. | O(n^3) scaling — slows down as training data grows beyond a few thousand samples. |

## Examples

**Enable with default settings (Random Forest):**
```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true
```

**Use Gaussian Process backend:**
```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true surrogate.model_type.value=gaussian_process
```

**Evaluation mode (predict but train everything, for benchmarking):**
```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true surrogate.evaluation_mode.value=true
```

**Tune the skip aggressiveness:**
```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true \
  surrogate.confidence_threshold.value=0.4 \
  surrogate.exploration_ratio.value=0.3
```

## Output Files

When the surrogate is enabled, the following files are written to `{results_dir}/surrogate/`:

| File | Contents |
|------|----------|
| `surrogate_log.csv` | Per-individual records: `generation, individual, predicted_acc, uncertainty, actual_acc, skipped` |
| `surrogate_summary.csv` | Per-generation aggregates: `generation, n_total, n_skipped, n_trained, mae, correlation, r_squared` |
| `surrogate_evaluation.png` | 4-panel plot (predicted vs actual, correlation over time, MAE over time, error distribution). Overwritten each generation. |
| `model.joblib` | Serialized scikit-learn model (for continue runs). |
| `training_data.npz` | Collected (encoding, accuracy) pairs. |
| `metadata.json` | Model configuration and state. |
