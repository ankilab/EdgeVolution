import time

import serial.tools.list_ports
import json
import numpy as np
import argparse


def get_nrf_port(board, timeout_in_ms = 2000):
    """ 
    get_nrf_port waits until the serial port for the corresponding board is available and return it after that.

    :param board: dict containing information about board {"model": <board_model>, "snr": <snr_as_string>}
    :param timeout_in_ms: timeout in ms 

    :return: port device if corresponding device is found

    :raises: RuntimeError if there corresponding device is not found   
    """ 
    
    # set up counter variables
    max_iterations_counter = 0

    # poll every 100 ms
    sleep_time_in_ms = 100

    # max_iterations x sleep_time = total_timeout
    max_iterations = int(timeout_in_ms // sleep_time_in_ms)
    
    while max_iterations_counter<max_iterations:
        try:
            # get ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # find port that contains <model> and <snr>
                if board["model"] in port.description:
                    if board["snr"] in port.description:
                        return port.device
            time.sleep(sleep_time_in_ms/1000)
        except TypeError:
            pass
        except Exception as e:
            print(f'caught {type(e)}: e')
        max_iterations_counter += 1
    raise RuntimeError("Could not find the port to listen to")

def read_inference_time(board, save_dir=None):
    port = None
    try:
        port = get_nrf_port(board)
    except RuntimeError as e:
        print(str(e))

    measured_values = []

    if port:
        max_iterations_counter = 0
        max_iterations = 40
        with serial.Serial(port, 115200, timeout=0) as ser:
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
    else:
        measured_values.append("Could not find the port of board. Please connect board. ")

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
        description='This script measures the inference speed of a neural network on the specified microcontroller.')

    parser.add_argument('save_dir', nargs='?', default=None)
    # board should be a dict specifying board model and snr e.g. {"model" : "nrf52840dk_nrf52840", "snr": "1050242564"}
    parser.add_argument('board_model', nargs='?', default=None)
    parser.add_argument('board_snr', nargs='?', default=None)

    args = parser.parse_args()

    time.sleep(2)
    board = {
        "model" : args.board_model,
        "snr" : args.board_snr
    }
    read_inference_time(board, args.save_dir) 
    time.sleep(2)
