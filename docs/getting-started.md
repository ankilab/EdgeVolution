# Getting Started with EdgeVolution

This guide will help you set up and run EdgeVolution on your system.

## Setup

### Cloning the repository

```bash
git clone https://github.com/ankilab/EdgeVolution
```

### Docker Build Targets

The Dockerfile provides two build targets. The `embedded` image extends `ml` — it contains everything in `ml` plus the hardware toolchain:

```
┌──────────────────────────────────────────────────────────────┐
│  embedded   (docker build --target embedded .)               │
│                                                              │
│  nRF CLI tools, Segger J-Link, Zephyr SDK, west workspace   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ml   (docker build .)              ← DEFAULT target   │  │
│  │                                                        │  │
│  │  TensorFlow 2.9.1-gpu · Python deps · project code    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**ML/NAS only** (default) — architecture search, training, evaluation:
```bash
docker build -t edgevolution .
```

**Embedded** — everything above plus cross-compilation and flashing to MCUs:
```bash
docker build --target embedded -t edgevolution-embedded .
```

> The embedded build downloads the Zephyr SDK and modules (~1.5 GB).
> Docker caches this layer, so subsequent builds are fast.

### Running the Container

**ML/NAS (GPU-accelerated):**
```bash
docker run -it --rm --gpus all -v $(pwd):/EdgeVolution edgevolution
```

**Embedded (with USB passthrough for J-Link):**
```bash
docker run -it --rm --privileged --gpus all -v $(pwd):/EdgeVolution edgevolution-embedded
```

## Configuration

### Defining hyperparameters
Before the EdgeVolution optimization run is started, some configurations must be made. These include the definition of hyperparameters, the search space and the boards that are to be used for evaluating the candidates on the microcontroller.

1. **Search space setup**  
    [Search Space](search-space.md)

2. **Hyperparameters**  
    [Hyperparameters](hyperparameters.md) --> Important to update results path!

3. **Microcontroller setup**  
    [Microcontrollers](microcontrollers.md)

4. **Dataloader definition**  
    [Dataloader Definition](dataloader-definition.md)

## Execution

### Running an experiment

```bash
python main.py +hyperparameters=<your_hyperparams> +search_space=<your_search_space> +boards=nrf52840dk
```

## Visualization

The [EvoVis Dashboard](evovis-dashboard.md) can be used to visualize and interpret the results of an EdgeVolution experiment run.