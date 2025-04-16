# Adding New Microcontrollers and Power Profiler Kits to the Project

This document provides instructions on how to add new microcontrollers and their corresponding Power Profiling Kits 2 (PPK2s) to the project.

## TL;DR
✅ Each microcontroller has a corresponding YAML file.

✅ The file contains details about the board model, serial number, and PPK2.

✅ To add a new microcontroller, create a new YAML file and update it with the correct details.

✅ Use `none.yaml` when no microcontroller constraints are required.

## Microcontroller Configuration Files

Each microcontroller has its own YAML configuration file (e.g., `nrf52840dk.yaml`, `nrf5340dk.yaml`). These files define the microcontroller model, serial number (SNR), and PPK2 serial number.

### YAML File Structure
Each YAML file should follow this structure:

```yaml
value:
   - model: "<board_model>"  # Board model
     snr: "<serial_number>"  # Serial number
     ppk: "<ppk_serial_number>"  # PPK (Power Profiling Kit) serial number
```

#### Example (`nrf52840dk.yaml`):
```yaml
value:
   - model: "nrf52840dk_nrf52840"  # Board model
     snr: "1050283223"  # Serial number
     ppk: "FAFD344C"  # PPK (Power Profiling Kit) serial number
```

### Adding a New Microcontroller
To add a new microcontroller:
1. Create a new YAML file in the same directory.
2. Name the file according to the microcontroller model (e.g., `nrf5340dk.yaml`).
3. Use the following template and update it with the correct details:

```yaml
value:
   - model: "<new_board_model>"
     snr: "<new_serial_number>"
     ppk: "<new_ppk_serial_number>"
```

4. If multiple configurations exist for the same model, you can add them as commented-out options for reference:

```yaml
# Alternative configurations:
# value:
#   - model: "nrf52840dk_nrf52840"
#     snr: "1050242564"
#     ppk: "FEA55411"
```

## Running Without a Microcontroller Constraint
If the optimization process should run without any microcontroller constraints, use the `none.yaml` file:

```yaml
value:
  - model: null
```

This file ensures that no specific microcontroller is enforced during execution.


