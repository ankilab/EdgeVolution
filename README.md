# EvoNAS

EvoNAS is a software-hardware end-to-end pipeline that allows to optimize preprocessing and neural network architectures for microcontrollers in a single step.

The neural architecture search (NAS) follows a evolutionary algorithm approach that optimizes accuracy, memory footprint, inference time and energy consumption deployed on the microcontroller. 

## Prerequisites

On Linux, the `install.sh` script should automatically install all the requirements. On Windows, make sure to manually install the following prerequisites:

* Python X.X installed

* You have installed the at least version 14.0 or greater of Microsoft Visual C++. Download [Installer](https://aka.ms/vs/17/release/vs_BuildTools.exe) (C++ x64/x86 build tools and Windows 10/11 SDK).

* Install xxd and add it to PATH. Downloaded [here](https://sourceforge.net/projects/xxd-for-windows/files/latest/download).

* Install ffprobe and ffmpeg and add them PATH. Download for example [here] (https://ffmpeg.org/download.html)

* Download the [nrf-toolchain](https://www.nordicsemi.com/Products/Development-tools/nrf-connect-for-desktop)

* Install west make sure west is accessible globally (not just in virtual env)
```
pip install west
```

## Installing EvoNAS

To install EvoNAS, follow these steps:

* Clone the github repository.
``` 
git clone https://github.com/ankilab/EvoNAS
```

### Linux
`install.sh` should automatically install all dependencies
```
cd ./EvoNAS
sudo ./install.sh
```
### Windows
For windows, manual installation is possible.

* Install requirements in virtual environment

```
# create virtual environment
python -m venv .env 

# activate virtual environment
.\.env\Scripts\activate

pip install -r requirements.txt
```

* Convert "flash_tflite_model.sh" to unix via wsl:
``` 
cd ./tools
dos2unix flash_tflite_model.sh
```

* Zephyr as described in wiki #TODO, but change protobuf version to 3.20.0 instead of 3.20.3

* Install zephyr SDK according to [this tutorial](https://docs.zephyrproject.org/3.2.0/develop/toolchains/zephyr_sdk.html). 
Manually download and extract SDK to C:/Program Files/zephyr-sdk-0.15.0 or in Powershell: 

```
cd "C:/Program Files"
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.15.0/zephyr-sdk-0.15.0_windows-x86_64.zip
unzip zephyr-sdk-0.15.0_windows-x86_64.zip
```

Then:
```
cd "C:/Program Files/zephyr-sdk-0.15.0"
./setup.sh

```

## Using EvoNAS

To use EvoNAS, follow these steps:

```
python main.py
```


## Contributing to EvoNAS
To contribute to EvoNAS, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


## Contact

If you want to contact me you can create a github issue.

## License

This project is still in development. A license still needs to be added. TODO: add license