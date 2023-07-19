#!/bin/bash
set -e

cd "$(dirname "$0")"

path_tflite=$1
path_cpp=$2
board_type=$3
board_snr=$4
source_folder=$5
name="$board_type-$board_snr"

python3 convert_to_c_array_and_insert_into_cpp.py $path_tflite $path_cpp

cd "../tflite"


# add the board type and snr to the config files as port description
# Read the project file and replace the CONFIG_USB_DEVICE_PRODUCT value
project_file_path="./$source_folder/prj.conf"
sed -i "s/CONFIG_USB_DEVICE_PRODUCT=\"[^\"]*\"/CONFIG_USB_DEVICE_PRODUCT=\"$name\"/" "$project_file_path"
echo "Value of CONFIG_USB_DEVICE_PRODUCT changed to '$name'"

# build the files for the given board and snr
west build -b $board_type $source_folder/ --build-dir build-$board_type

# flash the given board
west flash --recover --dev-id $board_snr --build-dir build-$board_type
