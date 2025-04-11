# Search Space Configuration File Structure

This document describes the structure and configuration of the YAML file that defines the search space for supported layers and their parameters. To extend the search space, follow the instructions outlined in the README.md file.

## TL;DR
✅ The YAML configuration defines the available layers and their parameters.

✅ Parameters using lists follow the `[start, stop, step]` format.

✅ Rule sets ensure valid model architectures.

✅ Extend the search space by adding layers in `layer_definitions.py` and updating the YAML.

## Extending the Search Space
To add a new layer to the search space:
1. Implement the layer in `neural_architecture_search/src/layer_definitions.py`.
2. If needed, place custom operations in `neural_architecture_search/src/search_space_modules/`.
3. Add the layer configuration to the `gene_pool` section of the YAML file.
4. Update the `rule_set` to define layer connectivity.

## Gene Pool
The `gene_pool` section defines available layers and their parameters for feature extraction, global pooling, preprocessing, and dense layers. All list-based parameters follow the `[start, stop, step]` format.

### Feature Extraction - 1D

| Layer Type | Function Name          | Filters/Kernel/Pool Size               | Strides   | Padding | Activation |
| ---------- | ---------------------- | -------------------------------------- | --------- | ------- | ---------- |
| `C_1D`     | `Conv1D`               | Filters: [4, 32, 1], Kernel: [1, 5, 1] | [1, 2, 1] | `same`  | `relu`     |
| `DC_1D`    | `DepthwiseConv1D`      | Kernel: [1, 5, 1]                      | [1, 2, 1] | `same`  | `relu`     |

### Feature Extraction - 2D

| Layer Type | Function Name          | Filters/Kernel/Pool Size               | Strides   | Padding | Activation |
| ---------- | ---------------------- | -------------------------------------- | --------- | ------- | ---------- |
| `C_2D`     | `Conv2D`               | Filters: [4, 32, 1], Kernel: [1, 5, 1] | [1, 2, 1] | `same`  | `relu`     |
| `DC_2D`    | `DepthwiseConv2D`      | Kernel: [1, 5, 1]                      | [1, 2, 1] | `same`  | `relu`     |

### Global Pooling

| Layer Type | Function Name              |
| ---------- | -------------------------- |
| `GAP_1D`   | `GlobalAveragePooling1D()` |
| `GMP_1D`   | `GlobalMaxPooling1D()`     |

### Preprocessing - 2D

| Layer Type   | Function Name          | Parameters                                           |
| ------------ | ---------------------- | ---------------------------------------------------- |
| `STFT_2D`    | `STFT`                 | `n_fft`: [64, 1024, 16], `hop_length`: [64, 396, 16] |

### Dense Layers

| Layer Type | Function Name | Parameters                               |
| ---------- | ------------- | ---------------------------------------- |
| `D`        | `Dense`       | Units: [8, 128, 8], Activation: [`relu`] |

## Rule Set

The `rule_set` defines valid layer connectivity.

### 1D Layer Rules

| Layer   | Next Possible Layers                                  |
| ------- | ----------------------------------------------------- |
| `C_1D`  | `AP_1D`, `MP_1D`, `BN_1D`, `DC_1D`, `C_1D`, `STFT_2D` |

### 2D Layer Rules

| Layer   | Next Possible Layers                                  |
| ------- | ----------------------------------------------------- |
| `STFT_2D` | `MAG_2D` |


