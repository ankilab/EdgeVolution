#!/bin/bash
set -e

path_tflite=$1
path_cpp=$2
board_type=$3
board_snr=$4
name="$board_type-$board_snr"

# Use EDGEVOLUTION_ROOT if set, otherwise derive from script location
PROJECT_DIR="${EDGEVOLUTION_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Backport argparse.BooleanOptionalAction for Python < 3.9 (needed by Zephyr west commands)
export PYTHONPATH="$PROJECT_DIR/tools/_sitecustomize:${PYTHONPATH:-}"

python3 "$PROJECT_DIR/tools/convert_to_c_array_and_insert_into_cpp.py" "$path_tflite" "$path_cpp"

cd "$PROJECT_DIR/tflite"

# Register Zephyr CMake package (idempotent, needed after fresh container start)
west zephyr-export > /dev/null 2>&1 || true

# add the board type and snr to the config files as port description
# Read the project file and replace the CONFIG_USB_DEVICE_PRODUCT value
project_file_path="./edgevolution_tflite/prj.conf"
sed -i "s/CONFIG_USB_DEVICE_PRODUCT=\"[^\"]*\"/CONFIG_USB_DEVICE_PRODUCT=\"$name\"/" "$project_file_path"
echo "Value of CONFIG_USB_DEVICE_PRODUCT changed to '$name'"

# build the files for the given board and snr
west build -b $board_type edgevolution_tflite/ --build-dir build-$board_type  #> /dev/null

# generate RAM report (ROM report skipped — ROM is static for a given build config)
echo "Generating RAM report"
west build --build-dir build-$board_type -t ram_report > /dev/null

# flash the given board
# Only recover if the device is AP-protected (avoid unnecessary USB re-enumeration)
if ! nrfjprog --snr $board_snr --eraseall 2>/dev/null; then
    echo "Erase failed — attempting recover (AP protection may be enabled)"
    nrfjprog --snr $board_snr --recover
    nrfjprog --snr $board_snr --eraseall
fi
west flash --dev-id $board_snr --build-dir build-$board_type  > /dev/null
