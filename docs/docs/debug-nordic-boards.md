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
* Open folder EdgeVolution in VS Code
* Run nRF Connect extension in VS Code
* On the WELCOME tab, click on `Open an existing application`
* Select ./tflite/edgevolution_tflite
* On the APPLICATIONS tab, click on the second icon next to edgevolution_tflite which adds a build configuration `Open an existing application`
* Select which board type to debug, use `prj.conf` as configuration and keep build as directory name. (NOTE: choosing a build directory outside of edgevolution_tflite via ../ does not save the build configuration)
* A build should have been started. The board can now be debugged using the Debug button in the ACTIONS tab. Make sure that the board appears in the CONNECTED DEVICES tab. When hooked up to the power profiler, the power profiler needs to be lit in blue in order to flash the kit properly.

### Workaround
For some reason, the nrf connect plugin does not include the modules located in tflite/modules. It seems to locate the the wrong zephyr base (not the one in ./tflite/zephyr), even though it is set. When copying the build command of the nRF Connect VS Code extension by right clicking the build in the APPLICATIONS tab, and manually running the command, it seems to run fine. Build the source then via this command, e.g.:
```
west build --build-dir /home/<user>/EvoNAS_bump/tflite/build /home/<user>/EvoNAS_bump/tflite --pristine --board nrf52840dk_nrf52840 -- -DNCS_TOOLCHAIN_VERSION:STRING="NONE" -DDTC_OVERLAY_FILE:STRING="/home/<user>/EvoNAS_bump/tflite/app.overlay" -DCONF_FILE:STRING="/home/<user>/EvoNAS_bump/tflite/prj.conf"
```
Now, you can use the Debug under ACTIONS in order to create a launch.json. In the Debug tab of VS Code, now this launch json can be selected to properly debug the code.