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


## Running Tests

```bash
pytest tests/ -v
```


## Configuration Reference

All config files live under `conf/`. See the READMEs in each subdirectory for details:

- [Hyperparameters](../conf/hyperparameters/README.md) — training parameters, population sizes, fitness weights
- [Search space](../conf/search_space/README.md) — layer types, parameter ranges, topology rules
- [Boards](../conf/boards/README.md) — MCU target definitions (`none` disables hardware evaluation)
