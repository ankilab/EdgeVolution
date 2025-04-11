# Dataset Configuration File Structure

This document provides instructions on how to structure and configure the YAML file for each dataset. The configuration file contains general parameters, dataset-specific settings, neural network training hyperparameters, evolutionary search parameters, and fitness evaluation criteria.

## TL;DR
✅ The YAML configuration defines dataset parameters and hyperparameters.

✅ The tables provide an overview of each parameter and its purpose.

✅ Example YAML snippets illustrate proper formatting.

✅ Decay lists dynamically adjust hyperparameters across generations to optimize the search process.


## General Parameters
These parameters define general settings that apply to the optimization and hardware constraints.

| Parameter              | Value Type | Description |
|------------------------|------------|-------------|
| `min_free_space_gpu`  | Integer    | Minimum GPU memory (in bytes) required for execution. |

Example:
```yaml
min_free_space_gpu:
  value: 6_000_000
```

## Dataset/DNN Training Hyperparameters
These parameters define dataset-related and deep neural network training configurations.

| Parameter       | Value Type  | Description |
|----------------|------------|-------------|
| `results_path` | String     | Directory where training results are stored. |
| `dataset_name` | String     | Name of the dataset used for training. |
| `sample_rate`  | Integer    | Sample rate of the dataset (e.g., in Hz). |
| `input_shape`  | List       | Input shape of the data. |
| `num_classes`  | Integer    | Number of output classes. |
| `top_activation` | String   | Activation function for the output layer. |
| `num_epochs`   | Integer    | Number of training epochs. |
| `batch_size`   | Integer    | Batch size for training. |
| `optimizer`    | String     | Optimizer used for training. |
| `loss`         | String     | Loss function used in training. |
| `metrics`      | List       | Evaluation metrics used during training. |

Example:
```yaml
results_path:
  value: "Results/"
dataset_name:
  value: "speech_commands"
sample_rate:
  value: 6000
input_shape:
  value: [6000, 1]
num_classes:
  value: 12
top_activation:
  value: "softmax"
num_epochs:
  value: 3
batch_size:
  value: 128
optimizer:
  value: "adam"
loss:
  value: "categorical_crossentropy"
metrics:
  value: ["accuracy"]
```

## EdgeVolution Hyperparameters
These parameters define the settings for evolutionary neural architecture search (EdgeVolution). The decay lists specify how hyperparameters change across generations to balance exploration and exploitation.

| Parameter                     | Value Type | Description |
|--------------------------------|------------|-------------|
| `num_generations`             | Integer    | Total number of generations for the search process. |
| `population_size_decay`       | List       | Defines how the population size reduces over generations. |
| `num_best_models_crossover_decay` | List | Determines how many top models participate in crossover as generations progress. |
| `mutation_rate_decay`         | List       | Specifies the mutation rate adjustments at different generations. |
| `max_num_feature_layers`      | Integer    | Maximum number of feature extraction layers allowed. |
| `max_num_classification_layers` | Integer  | Maximum number of classification layers allowed. |

**Explanation of Decay Lists:**
- Each list consists of tuples in the format `[(generation, value), ...]`.
- The value takes effect from the specified generation onwards.
- This allows gradual tuning of parameters as training progresses.

Example:
```yaml
num_generations:
  value: 30
population_size_decay:
  value: [[1, 500], [2, 250], [10, 100]]
num_best_models_crossover_decay:
  value: [[1, 100], [2, 50], [10, 20]]
mutation_rate_decay:
  value: [[1, 30], [2, 25], [5, 20], [10, 15]]
max_num_feature_layers:
  value: 8
max_num_classification_layers:
  value: 4
```

## Fitness Evaluation Criteria
These parameters define the fitness function used in model selection.

| Parameter                | Value Type | Description |
|--------------------------|------------|-------------|
| `min_rom_usage`         | Integer    | Minimum required ROM usage in bytes. |
| `min_energy_information` | Float      | Minimum energy efficiency required. |
| `acc_weight`            | Float      | Weight for accuracy in fitness evaluation. |
| `rom_usage_weight`      | Float      | Weight for ROM usage in fitness evaluation. |
| `energy_information_weight` | Float  | Weight for energy efficiency in fitness evaluation. |

Example:
```yaml
min_rom_usage:
  value: 150_000
min_energy_information:
  value: 0.5
acc_weight:
  value: 0.7
rom_usage_weight:
  value: 0.1
energy_information_weight:
  value: 0.2
```

