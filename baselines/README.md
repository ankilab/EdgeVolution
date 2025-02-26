# EfficientNet Baseline Training

This repository provides a script to train an EfficientNet model on different datasets. The baseline training setup can be reproduced by running the script `train_efficientnet_baseline.py`.

## Running the Baseline

To train the baseline model, run the following command:

```sh
python train_efficientnet_baseline.py --dataset <dataset_name>
```

## Available Datasets

- `speech_commands`(default)
- `emg_airob`
- `cifar10` 
- `daliac`

## Creating baseline for each of the aforementioned datasets by using 'all'

```sh
python train_efficientnet_baseline.py --dataset all
```
