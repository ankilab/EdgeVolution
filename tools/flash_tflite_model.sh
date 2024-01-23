#!/bin/bash
set -e

cd "$(dirname "$0")"

path_tflite=$1
path_cpp=$2
board_type=$3
board_snr=$4
name="$board_type-$board_snr"

python3 convert_to_c_array_and_insert_into_cpp.py $path_tflite $path_cpp

cd "../tflite"

# add the board type and snr to the config files as port description
# Read the project file and replace the CONFIG_USB_DEVICE_PRODUCT value
project_file_path="./evonas_tflite/prj.conf"
sed -i "s/CONFIG_USB_DEVICE_PRODUCT=\"[^\"]*\"/CONFIG_USB_DEVICE_PRODUCT=\"$name\"/" "$project_file_path"
echo "Value of CONFIG_USB_DEVICE_PRODUCT changed to '$name'"

# build the files for the given board and snr
west build -b $board_type evonas_tflite/ --build-dir build-$board_type

# generate RAM and ROM reports
#west build --build-dir build-$board_type -t ram_report
west build --build-dir build-$board_type -t rom_report > /dev/null

# flash the given board
nrfjprog --eraseall  # this is specific to the nrf boards
west flash --recover --dev-id $board_snr --build-dir build-$board_type
