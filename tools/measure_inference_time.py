import time

import serial.tools.list_ports
import json
import numpy as np
import argparse


def get_nrf_port():
    """ Wait until the serial port is available and return it after that. """
    while True:
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if "Zephyr" in port.description:
                    return port.device
            time.sleep(0.1)
        except:
            pass


def read_inference_time(save_dir=None):
    port = get_nrf_port()
    measured_values = []

    max_iterations_counter = 0
    max_iterations = 200
    with serial.Serial(port, 115200, timeout=1) as ser:
        while len(measured_values) < 1:
            time.sleep(0.1)
            try:
                line = ser.readline()
                if line != b'':
                    if "AllocateTensor" in str(line) or "failed" in str(line) or "error" in str(line) or "exit" in str(line):
                        measured_values.append(str(line))
                    elif "InfTime" in str(line):
                        inf_time = int(line[9:-3:])
                        measured_values.append(inf_time)
                    else:
                        print(str(line))
                if max_iterations_counter > max_iterations:
                    measured_values.append("Max iterations reached")
                else:
                    max_iterations_counter = max_iterations_counter + 1
            except Exception as e:
                print(str(e))
                pass

    if save_dir is not None:
        # save the measured value to results.json
        with open(save_dir + '/results.json') as f:
            d = json.loads(f.read())
        try:
            d["inference_time"] = np.mean(measured_values)
        except:
            d["inference_time"] = measured_values
        with open(save_dir + '/results.json', 'w') as f:
            json.dump(d, f, indent=2)
    else:
        print(measured_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MeasureInferenceTime',
        description='This script measures the inference speed of a neural network on the NRF52840 microcontroller.')

    parser.add_argument('save_dir', nargs='?', default=None)
    args = parser.parse_args()

    time.sleep(2)
    read_inference_time(args.save_dir)
    time.sleep(2)
