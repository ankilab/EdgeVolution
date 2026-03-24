# Hardware Lookup Table (LUT)

A Hardware LUT is a pre-built cost model for a specific MCU target.  It is built from a **profiling phase** that evaluates random architectures on real hardware, and then replaces real MCU evaluation during NAS.

## Motivation

Real MCU evaluation (flash + measure inference time + measure energy) takes 30–60 seconds per architecture.  In a 30-generation NAS run with 50 individuals per generation, this adds up to ~12–25 hours of hardware time.

The Hardware LUT addresses this by:
1. **Profile once**: Evaluate N random architectures on the target board.
2. **Build LUT**: Train a cost model from the profiling data.
3. **Run NAS**: Use the LUT instead of real hardware — no board needed.
4. **Reuse**: The same LUT works for any future NAS run targeting the same board + search space.

### Difference from the online hardware surrogate

| | Online Surrogate (`surrogate_hardware`) | Hardware LUT |
|---|---|---|
| **When** | Learns during NAS run | Dedicated profiling phase before NAS |
| **Coverage** | Biased by NAS selection pressure | Uniform random sampling of search space |
| **Hardware needed during NAS** | Yes (exploration set always evaluated on MCU) | No |
| **Reusable** | Coupled to one run | Portable across runs |

## Prediction Modes

### Full-architecture mode (`full`)

One Random Forest per metric (energy, inference_time, ROM) trained on the full 544-dimensional architecture encoding.

- Captures cross-layer interactions (memory layout, caching effects).
- Works best when profiling data covers the search space well.

### Layer-wise mode (`layerwise`)

Per-layer-type regressors where total cost = sum of per-layer costs.

- Better generalization to unseen architecture structures.
- Interpretable: shows which layer types dominate the cost.
- Use `predict_breakdown()` to get per-layer cost contributions.

Training approach:
1. Extract per-layer features from the encoding's slot-based structure.
2. Train one Random Forest per layer type (C_2D_BLOCK, DC_2D_BLOCK, D, etc.) per metric.
3. Calibrate with a global scaling factor so sum of layer predictions matches measured totals.

## Workflow

### 1. Profile

Run the profiling script from Docker (requires MCU hardware):

```bash
docker run -it --rm --privileged --gpus all \
    -v /path/to/EdgeVolution:/EdgeVolution edgevolution-embedded \
    python3 tools/profile_hardware.py \
        +hyperparameters=speech_commands +search_space=speech_commands \
        +boards=nrf52840dk \
        hardware_profile.n_samples=200 \
        hardware_profile.output=hardware_luts/nrf52840dk/ \
        hardware_profile.mode=full
```

This will:
1. Generate 200 random architectures.
2. Evaluate them through the full pipeline (translate, train, flash, measure).
3. Build the LUT and report cross-validation R²/MAE per metric.
4. Save the LUT to the specified output directory.

### 2. Run NAS with LUT

No hardware needed:

```bash
python main.py \
    +hyperparameters=speech_commands +search_space=speech_commands \
    +boards=nrf52840dk search_strategy=pymoo \
    hardware_lut.enabled.value=true \
    hardware_lut.path.value=hardware_luts/nrf52840dk/
```

Results will contain `hw_lut_predicted: true` in each individual's `results.json`.

### 3. Reuse

The same LUT works for any future NAS run targeting the same board and search space combination.

## Configuration

In `conf/config.yaml`:

```yaml
hardware_lut:
  enabled:
    value: false
  path:
    value: null

hardware_profile:
  n_samples:
    value: 200
  output:
    value: null
  mode:
    value: full   # 'full' or 'layerwise'
```

## LUT Directory Structure

```
hardware_luts/nrf52840dk/
  lut_metadata.json       # board info, sample count, mode
  registry_info.json      # encoding schema
  energy/                 # per-metric model
    metadata.json         # (full mode: SurrogateModel format)
    model.joblib
    training_data.npz
  inference_time/
    ...
  rom/
    ...
```

## API

```python
from neural_architecture_search.src.hardware_lut import HardwareLUT

# Build from profiling results
lut = HardwareLUT.build_from_results(
    results_dir, board_snr, registry, mode="full"
)

# Predict
pred = lut.predict(chromosome, encoding)
# {'energy': (35.2, 2.1), 'inference_time': (2816.0, 150.3), 'rom': (489000.0, 12000.5)}

# Layer-wise breakdown (layerwise mode only)
breakdown = lut.predict_breakdown(chromosome, encoding)
# {'energy': {'0_STFT_2D': 5.1, '1_C_2D_BLOCK': 12.3, ...}}

# Cross-validation
cv = lut.cross_validate(results_dir, board_snr, registry, n_folds=5)
# {'energy': {'r2': 0.92, 'mae': 2.1}, ...}

# Save / Load
lut.save("hardware_luts/nrf52840dk/")
lut = HardwareLUT.load("hardware_luts/nrf52840dk/")
```
