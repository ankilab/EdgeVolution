![Logo](images/logo.png)

EdgeVolution is a software-hardware end-to-end pipeline that allows to optimize preprocessing and neural network architectures for microcontrollers in an end-to-end fashion. It features different building blocks that can be customized individually.

![Overview](images/overview.png)


# Getting Started

Welcome to EdgeVolution! You can follow this simple guide here to get a high-level overview. You can find a more comprehensive documentation [here](https://ankilab.github.io/EdgeVolution/).

### Cloning the repository

``` 
git clone https://github.com/ankilab/EdgeVolution
```

### Building the Docker Image

EdgeVolution provides two Docker build targets:

**ML/NAS target** (default) — For running neural architecture search and training:
```bash
docker build -t edgevolution .
```

**Embedded target** — For flashing optimized models to MCU hardware (includes nRF tools, J-Link, Zephyr SDK):
```bash
docker build --target embedded -t edgevolution-embedded .
```

> **Note:** The embedded target downloads the Zephyr SDK and modules (~1.5 GB) during the build. This is cached for subsequent builds.

### Running the Container

**ML/NAS (GPU-accelerated):**
```bash
docker run -it --rm --gpus all -v $(pwd):/EdgeVolution edgevolution
```

**Embedded (with USB device passthrough for J-Link):**
```bash
docker run -it --rm --privileged --gpus all -v $(pwd):/EdgeVolution edgevolution-embedded
```

### Running Tests

Inside the container:
```bash
pytest tests/ -v
```

### Defining hyperparameters
Before the EdgeVolution optimization run is started, some configurations must be made. These include the definition of hyperparameters, the search space and the boards that are to be used for evaluating the candidates on the microcontroller.

1) Search space setup
[search space setup](conf/search_space/README.md)

2) Hyperparameters
[hyperparameters](conf/hyperparameters/README.md)  --> Important to update results path!

3) Microcontroller setup
[microcontroller boards](conf/boards/README.md)


### Running an experiment

```
python main_edgevolution.py +hyperparameters=speech_commands +search_space=speech_commands +boards=none
```

For more examples and common commands, see the [Usage Guide](docs/usage.md).

# Community Support
Community support is provided via a [Discord Server](https://discord.gg/dAVbR923cD) for real-time community support.


# Contributing to EdgeVolution
To contribute to EdgeVolution, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


# Contact

If you want to contact me you can create a github issue or reach out on Discord.

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

# Citation

tbd
