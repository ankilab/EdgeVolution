#!/bin/sh
set -e

#conda init bash
#python3 src/convert-h5-to-tflite.py

#xxd -i airway_model.tflite > airway_model.cc
#rm airway_model.tflite
#python3 src/copy_c_array_to_cpp_file.py
#rm airway_model.cc

arduino-cli compile --fqbn adafruit:nrf52:cplaynrf52840 TfLiteSketch
arduino-cli upload -p /dev/ttyACM0 --fqbn adafruit:nrf52:cplaynrf52840 TfLiteSketch


