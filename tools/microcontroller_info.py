#!/usr/bin/env python3

import sys
import os
import serial
import serial.tools.list_ports
import argparse
import json
import re
import yaml

def get_available_devices():
    """
    Get a list of all available serial devices/microcontrollers
    Returns:
        list: List of available device objects
    """
    return list(serial.tools.list_ports.comports())

def check_devices_available():
    """
    Check if any microcontrollers are connected
    Returns:
        bool: True if any devices are available, False otherwise
    """
    devices = get_available_devices()
    return len(devices) > 0

def print_device_details(device):
    """
    Print detailed information about a specific device
    """
    print(f"\nDevice: {device.device}")
    print(f"Description: {device.description}")
    print(f"Hardware ID: {device.hwid}")
    
    if device.manufacturer:
        print(f"Manufacturer: {device.manufacturer}")
    if device.product:
        print(f"Product: {device.product}")
    if device.serial_number:
        print(f"Serial Number: {device.serial_number}")
    if device.vid:
        print(f"Vendor ID: {device.vid:04x}")
    if device.pid:
        print(f"Product ID: {device.pid:04x}")
    if device.location:
        print(f"Location: {device.location}")
    if device.interface:
        print(f"Interface: {device.interface}")

def list_all_devices():
    """
    List all connected microcontroller devices with details
    """
    devices = get_available_devices()
    
    if not devices:
        print("No microcontrollers detected.")
        return False
    
    print(f"Found {len(devices)} device(s):")
    for i, device in enumerate(devices, 1):
        print(f"\n--- Device {i} ---")
        print_device_details(device)
        
    return True

def get_nordic_board_type(device):
    """
    Determine Nordic board type from device information
    """
    desc = device.description.lower() if device.description else ""
    manufacturer = device.manufacturer.lower() if device.manufacturer else ""
    product = device.product.lower() if device.product else ""
    
    # PPK2 detection
    if "ppk" in desc or "ppk" in product:
        return "ppk2"
    
    # J-Link detection
    if "segger" in manufacturer and "j-link" in product:
        return "jlink"
        
    # nRF device detection
    if "nordic semiconductor" in manufacturer or "nordic" in manufacturer:
        if "nrf52840" in desc or "nrf52840" in product:
            return "nrf52840dk"
        elif "nrf52833" in desc or "nrf52833" in product:
            return "nrf52833dk"
        elif "nrf5340" in desc or "nrf5340" in product:
            return "nrf5340dk"
        elif "nrf52" in desc or "nrf52" in product:
            return "nrf52dk"
        elif "nrf51" in desc or "nrf51" in product:
            return "nrf51dk"
        else:
            return "nordic"
    
    # Default to unknown
    return "unknown"

def is_nordic_device(device):
    """
    Check if the device is a Nordic microcontroller or related tool
    """
    manufacturer = device.manufacturer.lower() if device.manufacturer else ""
    desc = device.description.lower() if device.description else ""
    product = device.product.lower() if device.product else ""
    
    return (
        "nordic semiconductor" in manufacturer or
        "nordic" in manufacturer or
        "segger" in manufacturer and "j-link" in product or
        "nrf" in desc or "nrf" in product
    )

def get_snr_from_device(device):
    """
    Extract a stable serial number (SNR) from device information
    """
    # If serial number is available, use it
    if device.serial_number:
        return device.serial_number
        
    # Otherwise, construct SNR from location and VID/PID
    location_part = device.location.replace(':', '_') if device.location else "noloc"
    vid_part = f"{device.vid:04x}" if device.vid else "0000"
    pid_part = f"{device.pid:04x}" if device.pid else "0000"
    
    return f"{vid_part}_{pid_part}_{location_part}"

def get_board_model(board_type):
    """
    Get the full model name based on board type
    """
    models = {
        "nrf52833dk": "nrf52833dk_nrf52833",
        "nrf52840dk": "nrf52840dk_nrf52840",
        "nrf5340dk": "nrf5340dk_nrf5340_cpuapp",
        "ppk2": "ppk2"
    }
    
    return models.get(board_type, board_type)

def get_default_tensor_arena_size(board_type):
    """
    Get default tensor arena size based on board type
    """
    sizes = {
        "nrf52833dk": 28,
        "nrf52840dk": 90,
        "nrf5340dk": 340,
        "ppk2": 0  # Not applicable for PPK2
    }
    
    return sizes.get(board_type, 0)

def create_config_for_nordic_device(device, board_type=None):
    """
    Create a configuration dictionary for a Nordic device in the required format
    """
    # Determine board type if not provided
    if not board_type:
        board_type = get_nordic_board_type(device)
    
    # Get SNR for the device
    snr = get_snr_from_device(device)
    
    # Get the full model name
    model = get_board_model(board_type)
    
    # Create config data in the required format
    config = {
        "value": [
            {
                "model": model,  # Board model
                "snr": snr,  # Serial number
                "ppk": "",  # PPK (Power Profiling Kit) serial number (unit: N/A)
                "power_measurement_threshold": 4000,  # Default power consumption threshold (unit: mA)
                "max_available_tensor_arena_size": get_default_tensor_arena_size(board_type)  # Available tensor arena size * 1024
            }
        ]
    }
    
    return config

def save_config_to_file(config, output_dir, board_type):
    """
    Save configuration to a file in the given directory
    """
    # Create filename
    filename = os.path.join(output_dir, f"{board_type}.yaml")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write config file
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {filename}")
    return filename

def print_config(config):
    """
    Print configuration in YAML format to console
    """
    print("\n--- Configuration YAML ---")
    print(yaml.dump(config, default_flow_style=False))

def get_nordic_devices():
    """
    Get a list of connected Nordic devices
    """
    devices = get_available_devices()
    return [d for d in devices if is_nordic_device(d) and not d.device.startswith('/dev/ttyS')]

def interactive_mode():
    """
    Interactive mode for generating Nordic device configurations
    """
    print("\n=== Nordic Microcontroller Configuration Tool ===\n")
    
    # Show all available devices
    all_devices = get_available_devices()
    
    if not all_devices:
        print("No devices detected.")
        return 1
    
    # Display all connected devices
    print("=== All Connected Devices ===")
    for i, device in enumerate(all_devices, 1):
        print(f"\n--- Device {i} ---")
        print_device_details(device)
    
    # Step 1: Ask which microcontroller type is being used (excluding PPK2)
    print("\nWhich microcontroller type are you using?")
    print("1. nRF52833dk")
    print("2. nRF52840dk")
    print("3. nRF5340dk")
    
    board_type = None
    while True:
        board_choice = input("\nSelect microcontroller type (1-3): ")
        if board_choice == '1':
            board_type = "nrf52833dk"
            break
        elif board_choice == '2':
            board_type = "nrf52840dk"
            break
        elif board_choice == '3':
            board_type = "nrf5340dk"
            break
        else:
            print("Invalid choice. Please try again.")
    
    print(f"\nSelected microcontroller type: {board_type}")
    
    # Step 2: Ask which device it corresponds to
    print("\n=== Select the corresponding device ===")
    nordic_devices = [d for d in all_devices if is_nordic_device(d)]
    
    if not nordic_devices:
        print("No Nordic devices detected that could match your selection.")
        return 1
    
    # List devices with index numbers
    for i, device in enumerate(nordic_devices, 1):
        print(f"{i}. {device.description}")
        print(f"   S/N: {device.serial_number if device.serial_number else 'unknown'}")
        if device.manufacturer:
            print(f"   Manufacturer: {device.manufacturer}")
    
    # Get user selection for the target device
    target_idx = 0
    if len(nordic_devices) > 1:
        while True:
            try:
                target_idx = int(input(f"\nSelect which device corresponds to your {board_type} (1-{len(nordic_devices)}): ")) - 1
                if 0 <= target_idx < len(nordic_devices):
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    target_device = nordic_devices[target_idx]
    target_serial = target_device.serial_number if target_device.serial_number else "unknown"
    
    print(f"\nSelected device: {target_device.description} (S/N: {target_serial})")
    
    # Step 3: Ask which PPK2 should be used
    ppk_devices = [d for d in all_devices if get_nordic_board_type(d) == "ppk2"]
    ppk_serial = ""
    
    if ppk_devices:
        print("\n=== Available PPK2 Devices ===")
        for i, device in enumerate(ppk_devices, 1):
            serial_num = device.serial_number if device.serial_number else "unknown"
            print(f"{i}. {device.description} - S/N: {serial_num}")
        
        has_ppk = input("\nDo you want to use a PPK2 for power measurements? (y/n): ").lower().strip()
        
        if has_ppk == 'y' or has_ppk == 'yes':
            if len(ppk_devices) == 1:
                ppk_device = ppk_devices[0]
                ppk_serial = ppk_device.serial_number if ppk_device.serial_number else "unknown"
                print(f"Using PPK2 with S/N: {ppk_serial}")
            else:
                while True:
                    try:
                        ppk_idx = int(input(f"Select PPK2 device (1-{len(ppk_devices)}): ")) - 1
                        if 0 <= ppk_idx < len(ppk_devices):
                            ppk_device = ppk_devices[ppk_idx]
                            ppk_serial = ppk_device.serial_number if ppk_device.serial_number else "unknown"
                            print(f"Using PPK2 with S/N: {ppk_serial}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
    else:
        print("\nNo PPK2 devices detected.")
        custom_ppk = input("Do you want to enter a PPK2 serial number manually? (y/n): ").lower().strip()
        if custom_ppk == 'y' or custom_ppk == 'yes':
            ppk_serial = input("Enter PPK2 serial number: ").strip()
    
    # Step 4: Create configuration and return YAML
    config = create_config_for_nordic_device(target_device, board_type)
    
    # Set PPK serial if provided
    if ppk_serial:
        config["value"][0]["ppk"] = ppk_serial
    
    # Ask for custom tensor arena size if needed
    if board_type != "ppk2":
        print(f"\nDefault tensor arena size for {board_type}: {config['value'][0]['max_available_tensor_arena_size']}KB")
        custom_size = input("Enter custom tensor arena size in KB (or press Enter for default): ").strip()
        if custom_size and custom_size.isdigit():
            config["value"][0]["max_available_tensor_arena_size"] = int(custom_size)
    
    # Generate YAML for copy-paste
    print("\n=== Generated Configuration ===")
    print("Copy and paste the following into your configuration file:")
    print_config(config)
    
    # Ask if user wants to save the configuration
    save_option = input("\nDo you want to save this configuration to a file? (y/n): ").lower().strip()
    
    if save_option == 'y' or save_option == 'yes':
        # Find EdgeVolution root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        # Check if we're in the EdgeVolution directory structure
        if os.path.basename(script_dir) == "tools" and os.path.isdir(os.path.join(parent_dir, "conf")):
            output_dir = os.path.join(parent_dir, "conf", "boards")
            save_config_to_file(config, output_dir, board_type)
        else:
            print("Error: Unable to locate EdgeVolution conf directory.")
            print("Please copy and paste the configuration manually.")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Get information about connected Nordic microcontrollers')
    parser.add_argument('--check', action='store_true', help='Only check if Nordic devices are available')
    parser.add_argument('--list', action='store_true', help='List all available devices')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode (default if no arguments provided)')
    
    args = parser.parse_args()
    
    # Default to interactive mode if no arguments
    if len(sys.argv) == 1:
        return interactive_mode()
    
    if args.interactive:
        return interactive_mode()
    
    if args.check:
        nordic_devices = get_nordic_devices()
        if nordic_devices:
            print(f"Found {len(nordic_devices)} Nordic device(s).")
            return 0
        else:
            print("No Nordic microcontrollers detected.")
            return 1
    
    # List all devices
    if args.list or not (args.check or args.interactive):
        list_all_devices()
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
