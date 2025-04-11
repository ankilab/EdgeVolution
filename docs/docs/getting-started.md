# Getting Started with EdgeVolution

This guide will help you set up and run EdgeVolution on your system.

## Setup

### Cloning the repository

```bash
git clone https://github.com/ankilab/EdgeVolution
```

### Building the Docker Image

```bash
docker build -t edgevolution-container .
```

### Running the Container

```bash
docker run -it --rm --user $(id -u):$(id -g) --privileged --cpus="10.0" --gpus all -v $(pwd):/EdgeVolution edgevolution-container
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