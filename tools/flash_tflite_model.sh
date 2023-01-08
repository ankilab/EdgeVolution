#!/bin/bash
set -e

cd "$(dirname "$0")"

python3 convert_to_c_array_and_insert_into_cpp.py $1 '../tflite/airway_tflite/src/model.cpp'

cd "../tflite"
west build -b nrf52840dk_nrf52840 airway_tflite/
west flash
