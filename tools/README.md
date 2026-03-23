# EdgeVolution Tools

This directory contains various utility tools for the EdgeVolution project.

## Available Tools

// ...existing code...

### Microcontroller Info

`microcontroller_info.py` - Provides details about connected Nordic microcontrollers and helps generate configuration files.

Usage:
```bash
# Interactive mode (default if no arguments)
python microcontroller_info.py

# Explicitly run in interactive mode
python microcontroller_info.py --interactive
python microcontroller_info.py -i

# Only check if Nordic devices are available
python microcontroller_info.py --check

# List all connected devices with details
python microcontroller_info.py --list
```

The interactive mode provides a menu-driven interface to:
1. List all connected devices with detailed information
2. Generate configuration files for the following Nordic devices:
   - nRF52833dk
   - nRF52840dk  
   - nRF5340dk
   - PPK2
3. Choose to display the configuration in the console or save it to the conf/boards directory

Example Nordic board configuration file (YAML):
```yaml
value:
  - model: "nrf52840dk_nrf52840"  # Board model
    snr: "<YOUR_BOARD_SNR>"  # Serial number (find via `nrfjprog --ids`)
    ppk: "<YOUR_PPK_SNR>"  # PPK (Power Profiling Kit) serial number
    power_measurement_threshold: 4000  # Power consumption threshold (unit: mA)
    max_available_tensor_arena_size: 90  # Available tensor arena size * 1024
```
