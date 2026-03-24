# Search Space Registry

The `SearchSpaceRegistry` is the bridge between human-readable architecture definitions (YAML configs) and the fixed-length numeric vectors that optimisation algorithms like NSGA-II operate on. It handles:

- Loading search space configurations (unified or legacy YAML format)
- Encoding chromosomes (lists of layer dicts) to fixed-length vectors
- Decoding vectors back to chromosomes, optionally enforcing connectivity rules
- Generating random valid architectures

## Overview

```
YAML config ──> SearchSpaceRegistry ──> encode(chromosome) ──> vector (np.ndarray)
                                    <── decode(vector)     <── optimizer mutates vector
```

An **architecture** is represented in two forms:

| Form | Type | Used by |
|------|------|---------|
| **Chromosome** | `list[dict]` — one dict per layer, with `layer`, `f_name`, and parameter values | Model builder, training loop |
| **Vector** | `np.ndarray` of shape `(vector_size,)` with values in `[0, 1]` | Optimiser (crossover, mutation) |

The registry converts between the two.


## Worked Example

Consider a tiny search space with three layer types:

```yaml
categories:
  feature_extraction:
    successors: [feature_extraction, global_pooling]
  global_pooling:
    successors: [classification]
  classification:
    successors: [classification]
    terminal: true

start: [CONV]

layers:
  CONV:
    category: feature_extraction
    f_name: Conv2D
    filters: [8, 32, 8]          # discrete: [8, 16, 24, 32]
    activation: [relu, sigmoid]   # categorical
  GAP:
    category: global_pooling
    f_name: GlobalAveragePooling2D()
  DENSE:
    category: classification
    f_name: Dense
    units: [16, 48, 16]           # discrete: [16, 32, 48]
    activation: [relu, sigmoid]   # categorical
```

### 1. Encoding schema

The registry sorts layer names alphabetically (`CONV`, `DENSE`, `GAP`) and collects the *union* of all parameters across every layer:

| Parameter | Type | Vector size | Notes |
|-----------|------|-------------|-------|
| `activation` | categorical | 2 | one-hot: `[relu, sigmoid]` |
| `filters` | discrete | 1 | normalized index in `[8, 16, 24, 32]` |
| `units` | discrete | 1 | normalized index in `[16, 32, 48]` |

The per-slot layout is:

```
 ┌─── layer one-hot (4) ───┐ ┌── params (4) ──┐
 │ empty  CONV  DENSE  GAP │ │ act0 act1 filt units │
 └─────────────────────────┘ └────────────────────┘
         slot_size = 8
```

- **Layer one-hot** has `num_layers + 1 = 4` entries. Index 0 is reserved for "empty slot".
- **Parameters** are ordered alphabetically. Each layer only writes to the parameters it owns; the rest stay zero.

With `max_layers = 4`, the total vector size is `4 slots x 8 = 32`.

### 2. Encoding a chromosome

```python
chromosome = [
    {"layer": "CONV",  "f_name": "Conv2D",  "filters": 16, "activation": "relu"},
    {"layer": "CONV",  "f_name": "Conv2D",  "filters": 32, "activation": "sigmoid"},
    {"layer": "GAP",   "f_name": "GlobalAveragePooling2D()"},
    {"layer": "DENSE", "f_name": "Dense",   "units": 48,   "activation": "relu"},
]

vector = registry.encode(chromosome)
```

Slot-by-slot:

| Slot | Layer | one-hot (4) | activation (2) | filters (1) | units (1) |
|------|-------|-------------|----------------|-------------|-----------|
| 0 | CONV | `[0, 1, 0, 0]` | `[1, 0]` | `0.33` | `0.00` |
| 1 | CONV | `[0, 1, 0, 0]` | `[0, 1]` | `1.00` | `0.00` |
| 2 | GAP | `[0, 0, 0, 1]` | `[0, 0]` | `0.00` | `0.00` |
| 3 | DENSE | `[0, 0, 1, 0]` | `[1, 0]` | `0.00` | `1.00` |

How individual values are encoded:

- **Categorical** (`activation`): one-hot. `relu` = `[1, 0]`, `sigmoid` = `[0, 1]`.
- **Discrete** (`filters`): normalized index. `filters=16` is index 1 of `[8, 16, 24, 32]` → `1 / 3 = 0.33`. `filters=32` is index 3 → `3 / 3 = 1.0`.
- Params that a layer doesn't own (e.g. `units` for CONV, `filters` for DENSE) are left at `0.0`.

### 3. Decoding a vector

```python
chromosome = registry.decode(vector, enforce_rules=True)
```

Decoding reverses the process:

1. For each slot, check if it is empty (index 0 dominant or all values < 0.1).
2. Read the layer one-hot and pick the `argmax`. If `enforce_rules=True`, invalid layers are masked to `-inf` before the argmax (e.g. after CONV, only `[CONV, GAP]` are valid successors, so DENSE is masked out).
3. Decode parameters: categorical → argmax of one-hot; discrete → denormalize index and snap to closest value.
4. Stop early once an empty slot is encountered (after at least `early_stop_threshold` layers).

### 4. Why this matters for optimisation

The vector representation lets standard evolutionary operators work directly:

- **Crossover** blends two parent vectors → child inherits structure from both.
- **Mutation** nudges values → e.g. changing `filters` from `0.33` to `0.50` shifts from `16` to `24`.
- **Repair** decodes and re-encodes with `enforce_rules=True` → any illegal layer transition produced by crossover/mutation is corrected.


## Programmatic Usage

### Loading a search space

```python
from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry

# From a YAML file (auto-detects unified vs. legacy format)
registry = SearchSpaceRegistry.from_yaml(
    "conf/search_space/speech_commands.yaml",
    max_layers=12,
    validate=False,   # skip LayerRegistry check
)

print(f"Layer types:  {registry.layer_names}")
print(f"Vector size:  {registry.vector_size}")
print(f"Max layers:   {registry.max_layers}")
```

### Encode / decode roundtrip

```python
# Create a random valid architecture
chromosome = registry.create_random_chromosome()

# Encode to vector
vector = registry.encode(chromosome)            # shape: (vector_size,)

# Decode back (with connectivity rules enforced)
decoded = registry.decode(vector, enforce_rules=True)

# Layer types are preserved
assert [g["layer"] for g in chromosome] == [g["layer"] for g in decoded]
```

### Querying connectivity rules

```python
# Which layers can start an architecture?
registry.get_start_layers()          # e.g. ["STFT_2D"]

# Which layers can follow a given layer?
registry.get_successors("C_2D_BLOCK")  # e.g. ["C_2D_BLOCK", "DC_2D_BLOCK", ..., "GAP_2D"]

# Can this layer end an architecture?
registry.is_terminal("D")             # True (classification category)
```


## Integration with PyMOO

The `examples/pymoo_integration.py` module shows how to plug the registry into multi-objective optimisation with NSGA-II:

```python
from neural_architecture_search.examples.pymoo_integration import (
    NASProblem, ArchitectureSampling, ArchitectureRepair,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

problem   = NASProblem(registry)
algorithm = NSGA2(
    pop_size=50,
    sampling=ArchitectureSampling(registry),
    repair=ArchitectureRepair(registry),       # decode → enforce rules → re-encode
)

result = minimize(problem, algorithm, ("n_gen", 20), verbose=True)

# Pareto-optimal architectures
for vector in result.X:
    arch = registry.decode(vector, enforce_rules=True)
    print([g["layer"] for g in arch])
```

The three components:

| Class | Role |
|-------|------|
| `NASProblem` | Defines the optimisation problem: `n_var = vector_size`, objectives = `[-accuracy, size, latency]` |
| `ArchitectureSampling` | Generates the initial population via `create_random_chromosome()` + `encode()` |
| `ArchitectureRepair` | After crossover/mutation: `decode(enforce_rules=True)` → `encode()` to guarantee valid architectures |

Run the standalone demo:

```bash
python -m neural_architecture_search.examples.pymoo_integration --demo
```
