# Usage Guide

Quick reference for common EdgeVolution commands. See the [README](../README.md) for setup instructions.


## Docker

**Build the ML/NAS image (default):**
```bash
docker build -t edgevolution .
```

**Build the embedded image (includes nRF tools, J-Link, Zephyr SDK):**
```bash
docker build --target embedded -t edgevolution-embedded .
```

**Run the ML container (GPU-accelerated):**
```bash
docker run -it --rm --gpus all -v $(pwd):/EdgeVolution edgevolution
```

**Run the embedded container (with USB passthrough for J-Link):**
```bash
docker run -it --rm --privileged --gpus all -v $(pwd):/EdgeVolution edgevolution-embedded
```


## Running Experiments

EdgeVolution uses [Hydra](https://hydra.cc/) for configuration. Experiments require three config groups:

| Group | Flag | Available configs |
|-------|------|-------------------|
| Hyperparameters | `+hyperparameters=` | `speech_commands`, `cifar10`, `daliac`, `emg_airob` |
| Search space | `+search_space=` | `speech_commands`, `cifar10`, `daliac`, `emg_airob`, `complete` |
| Boards | `+boards=` | `none`, `nrf52840dk`, `nrf5340dk`, `nrf52833dk` |

### Speech commands (no MCU evaluation)

```bash
python main_edgevolution.py +hyperparameters=speech_commands +search_space=speech_commands +boards=none
```

### Speech commands with MCU evaluation on nRF52840

```bash
python main_edgevolution.py +hyperparameters=speech_commands +search_space=speech_commands +boards=nrf52840dk
```

### CIFAR-10 (no MCU evaluation)

```bash
python main_edgevolution.py +hyperparameters=cifar10 +search_space=cifar10 +boards=none
```

### DaLiAc (no MCU evaluation)

```bash
python main_edgevolution.py +hyperparameters=daliac +search_space=daliac +boards=none
```

### Override individual parameters

Hydra lets you override any config value from the command line:

```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands \
  +search_space=speech_commands \
  +boards=none \
  hyperparameters.num_epochs.value=10 \
  hyperparameters.num_generations.value=5
```


## Continuing a Run

```bash
python main_edgevolution.py \
  continue_path=Results/speech_commands/<run_folder> \
  continue_generation=5
```


## Surrogate-Assisted Search

EdgeVolution includes an optional surrogate model that predicts validation accuracy from architecture encodings. It pre-screens individuals each generation and skips training for those confidently predicted to perform poorly, reducing overall search time.

Two model backends are available:

| Backend | Flag value | Strengths |
|---------|-----------|-----------|
| Random Forest | `random_forest` (default) | Fast, robust, good default. Tree-variance provides uncertainty. |
| Gaussian Process | `gaussian_process` | Calibrated Bayesian uncertainty. Best for small datasets. O(n³) scaling. |

### Enable surrogate pre-screening

```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true
```

### Use a Gaussian Process backend

```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true surrogate.model_type.value=gaussian_process
```

### Evaluation mode (predict but still train everything)

In evaluation mode the surrogate predicts accuracy for every individual but never skips any — all individuals are still fully trained. This produces ground-truth predicted-vs-actual data useful for paper figures (scatter plots, error distributions, per-generation correlation).

```bash
python main_edgevolution.py \
  +hyperparameters=speech_commands +search_space=speech_commands +boards=none \
  surrogate.enabled.value=true surrogate.evaluation_mode.value=true
```

### Tuning the accuracy vs. time trade-off

The surrogate skips training for individuals it confidently predicts to perform poorly. This saves time but means those architectures never get a real evaluation — if the surrogate is wrong, good candidates may be discarded. Two parameters control this trade-off directly:

- **`confidence_threshold`** (default `0.5`) — Only individuals predicted *below* this accuracy are skip candidates. Lowering it makes the surrogate more conservative (skips fewer, wastes less potential). Raising it skips more aggressively and saves more time, but increases the risk of discarding promising architectures.
- **`exploration_ratio`** (default `0.2`) — Fraction of the population that is *always* trained, regardless of predictions. This prevents the surrogate from reinforcing its own biases. A higher ratio is safer but reduces the time savings; a lower ratio maximizes speedup at the cost of exploration.

As a rule of thumb: if your per-individual training time is **short** (a few seconds to minutes), the surrogate overhead may not be worth the risk — train everything. If training is **expensive** (tens of minutes to hours per individual), even a moderately accurate surrogate pays for itself by cutting the population that needs full training.

Start with evaluation mode (`surrogate.evaluation_mode.value=true`) to measure the surrogate's accuracy on your specific search space before relying on it to skip training. Check `surrogate_evaluation.png` in the results folder — if the correlation is low or the MAE is large relative to accuracy differences in your population, keep the confidence threshold conservative or increase the exploration ratio.

### Output files

When the surrogate is enabled, two CSV files are written to `{results_dir}/surrogate/`:

| File | Contents |
|------|----------|
| `surrogate_log.csv` | Per-individual records: `generation, individual, predicted_acc, uncertainty, actual_acc, skipped` |
| `surrogate_summary.csv` | Per-generation aggregates: `generation, n_total, n_skipped, n_trained, mae, correlation, r_squared` |


## Running Tests

```bash
pytest tests/ -v
```


## Configuration Reference

All config files live under `conf/`. See the READMEs in each subdirectory for details:

- [Hyperparameters](../conf/hyperparameters/README.md) — training parameters, population sizes, fitness weights
- [Search space](../conf/search_space/README.md) — layer types, parameter ranges, topology rules
- [Boards](../conf/boards/README.md) — MCU target definitions (`none` disables hardware evaluation)
- [Surrogate](surrogate.md) — surrogate model parameters, backend options, output files
