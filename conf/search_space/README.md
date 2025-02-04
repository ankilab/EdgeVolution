# Search Space Configuration File Structure

This document describes the structure and configuration of the YAML file that defines the search space for supported layers and their parameters. To extend the search space, follow the instructions outlined in the README.md file.

## Extending the Search Space
To add a new layer to the search space, follow these steps:
1. Implement the new layer in the `neural_architecture_search/src/layer_definitions.py` file.
2. If the layer requires custom operations, place the implementation in the `neural_architecture_search/src/search_space_modules/<your_custom_layer.py>` folder.
3. Add the new layer configuration to the `gene_pool` section of the YAML file.
4. Update the `rule_set` section to define how the new layer connects to other layers.

## Gene Pool
The `gene_pool` section defines the available layers and their parameters for both 1D and 2D feature extraction, global pooling, preprocessing, and dense layers.

### Feature Extraction - 1D

| Layer Type | Function Name          | Filters/Kernel/Pool Size               | Strides   | Padding | Activation |
| ---------- | ---------------------- | -------------------------------------- | --------- | ------- | ---------- |
| `C_1D`     | `Conv1D`               | Filters: [4, 32, 1], Kernel: [1, 5, 1] | [1, 2, 1] | `same`  | `relu`     |
| `DC_1D`    | `DepthwiseConv1D`      | Kernel: [1, 5, 1]                      | [1, 2, 1] | `same`  | `relu`     |
| `MP_1D`    | `MaxPooling1D`         | Pool Size: [2, 4, 1]                   | -         | `same`  | -          |
| `AP_1D`    | `AveragePooling1D`     | Pool Size: [2, 4, 1]                   | -         | `same`  | -          |
| `BN_1D`    | `BatchNormalization()` | -                                      | -         | -       | -          |

### Feature Extraction - 2D

| Layer Type | Function Name          | Filters/Kernel/Pool Size               | Strides   | Padding | Activation |
| ---------- | ---------------------- | -------------------------------------- | --------- | ------- | ---------- |
| `C_2D`     | `Conv2D`               | Filters: [4, 32, 1], Kernel: [1, 5, 1] | [1, 2, 1] | `same`  | `relu`     |
| `DC_2D`    | `DepthwiseConv2D`      | Kernel: [1, 5, 1]                      | [1, 2, 1] | `same`  | `relu`     |
| `MP_2D`    | `MaxPooling2D`         | Pool Size: [2, 4, 1]                   | -         | `same`  | -          |
| `AP_2D`    | `AveragePooling2D`     | Pool Size: [2, 4, 1]                   | -         | `same`  | -          |
| `BN_2D`    | `BatchNormalization()` | -                                      | -         | -       | -          |

### Global Pooling

| Layer Type | Function Name              |
| ---------- | -------------------------- |
| `GAP_1D`   | `GlobalAveragePooling1D()` |
| `GMP_1D`   | `GlobalMaxPooling1D()`     |
| `GAP_2D`   | `GlobalAveragePooling2D()` |
| `GMP_2D`   | `GlobalMaxPooling2D()`     |

### Preprocessing - 2D

| Layer Type   | Function Name          | Parameters                                           |
| ------------ | ---------------------- | ---------------------------------------------------- |
| `STFT_2D`    | `STFT`                 | `n_fft`: [64, 1024, 16], `hop_length`: [64, 396, 16] |
| `MAG_2D`     | `Magnitude()`          | -                                                    |
| `FB_2D`      | `get_filterbank_layer` | Type: [`mel`], `n_mels`: [32, 96, 4]                 |
| `MAG2DEC_2D` | `MagnitudeToDecibel()` | -                                                    |

### Dense Layers

| Layer Type | Function Name | Parameters                               |
| ---------- | ------------- | ---------------------------------------- |
| `D`        | `Dense`       | Units: [8, 128, 8], Activation: [`relu`] |
| `DO`       | `Dropout`     | Rate: [0.0, 0.5, 0.1]                    |

## Rule Set

The `rule_set` defines the connections between different layer types, ensuring valid model architectures are generated.

### 1D Layer Rules

| Layer   | Next Possible Layers                                  |
| ------- | ----------------------------------------------------- |
| `C_1D`  | `AP_1D`, `MP_1D`, `BN_1D`, `DC_1D`, `C_1D`, `STFT_2D` |
| `DC_1D` | `AP_1D`, `MP_1D`, `BN_1D`, `DC_1D`, `C_1D`, `STFT_2D` |
| `MP_1D` | `BN_1D`, `DC_1D`, `C_1D`, `STFT_2D`                   |
| `AP_1D` | `BN_1D`, `DC_1D`, `C_1D`, `STFT_2D`                   |

## Summary

✅ The YAML configuration file defines the search space for layers and their parameters.

✅ Layers are categorized into 1D, 2D, global pooling, preprocessing, and dense groups.

✅ Rule sets ensure valid layer connectivity for model architecture search.

✅ The tables in this file provide an overview of all currently available layers and their configurations.

✅ To extend the search space, implement new layers in `layer_definitions.py` and update the YAML configuration.
