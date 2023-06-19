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
Follow these steps for Linux:
* `install.sh` should automatically install all dependencies
    ```
    cd ./EvoNAS
    ./install.sh
    ```
* Add path variables and device permissions automatically.
  This manipulates the users device permissions and shell config files automatically. This script will tamper the users shell config files. Please run `./add_env_and_permissions.sh`
    
* (Optional) Add path variables and device permissions manually: 
    In order to properly use the toolchain, it is required to set the following PATH variables. Please add the following line to your .profile, .bashrc or other preferred shell configuration file: 
    ```export PATH="<path/to/add>:$PATH"" ```

    Next, add the permissions for the plugged in power profiler kits and the development kits:
    First, list all the idProduct of the kits having the idVendor (1366) of JLink programmer:
    ```
    lsusb |sed -n 's/.*1366:\([0-9]*\).*/\1/p'
    ```
    As *root* add a file named 50-nrf-access.rules to /etc/udev/rules.d. For every `<idProduct>` add the following line to this file:
    ```
    SUBSYSTEM=="usb", ATTRS{idVendor}=="1366", ATTRS{idProduct}=="<idProduct>", MODE="0777"
    ```
    Repeat the same for the power profiler kits using the idVendor 1915. If there are problems to get the ids, use `lsusb` to see all connected devices. It should output something like this:
    ```
    Bus 001 Device 007: ID 1915:c00a Nordic Semiconductor ASA PPK2
    ```

* Relogin to apply changes.

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

* Install zephyr SDK according to [this tutorial](https://docs.zephyrproject.org/3.2.0/develop/toolchains/zephyr_sdk.html). Manually download and extract SDK to C:/Program Files/zephyr-sdk-0.15.0 or in Powershell: 
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
## Debugging boards

To debug a NRF development kit board, follow these steps according this [tutorial](https://www.youtube.com/watch?v=zcMCaODyISo&list=PLx_tBuQ_KSqEt7NK-H7Lu78lT2OijwIMl&index=1):

* Download the nrf connect desktop tool [here](https://www.nordicsemi.com/Products/Development-tools/nrf-connect-for-desktop).
* Install the tool chain manager via the desktop tool
* Install the nRF Connect SDK via tool chain manager
* Install the VS Code extension by clicking the VS Code button next to the installed nRF Connect SDK in the tool chain manager
* Add `c_cpp_properties.json` to `.vscode` directory to include paths for autocompletion.
    ```
    {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/**"
                ],
                "defines": [],
                "compilerPath": "/usr/bin/gcc",
                "cStandard": "c17",
                "cppStandard": "gnu++17",
                "intelliSenseMode": "linux-gcc-x64"
            }
        ],
        "version": 4
    }
    ```
* Open folder EvoNAS in VS Code
* Run nRF Connect extension in VS Code
* On the WELCOME tab, click on `Open an existing application`
* Select ./tflite/airway_tflite 
* On the APPLICATIONS tab, click on the second icon next to airway_tflite which adds a build configuration `Open an existing application`
* Select which board type to debug, use `prj.conf` as configuration and keep build as directory name. (NOTE: choosing a build directory outside of airway_tflite via ../ does not save the build configuration)
* A build should have been started. The board can now be debugged using the Debug button in the ACTIONS tab. Make sure that the board appears in the CONNECTED DEVICES tab. When hooked up to the power profiler, the power profiler needs to be lit in blue in order to flash the kit properly. 

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
