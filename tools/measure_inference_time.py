import time

import serial.tools.list_ports
import json
import numpy as np
import argparse
import re

####################################################################################################
# This script measures the inference time of a neural network on a microcontroller.
####################################################################################################


def get_board_port(board, timeout_in_ms = 3000):
    """ 
    get_board_port waits until the serial port for the corresponding board is available and return it after that.

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
    
    while max_iterations_counter < max_iterations:
        try:
            # get ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # find port that contains <model> and <snr>
                if board["model"] in port.description:
                    if board["snr"] in port.description:
                        return port.device
            time.sleep(sleep_time_in_ms / 1000) # divide by 1000 to convert to seconds, because time.sleep expects seconds
        except TypeError:
            pass
        except Exception as e:
            print(f'caught {type(e)}: {e}')
        max_iterations_counter += 1
    raise RuntimeError("Could not find the port to listen to")


def set_result_value_for_board(board_snr, category, value, results):
    """ 
    update_or_set_result_value checks if the board information is already available and appends it to the dict
    structure of "<type>_information" is expected to be this:
    "<type>_information" : {
                "board_1" : [board_1_information],
                "board_2: : [board_2_information],
                ....
    }

    :param board_snr: string containing information the board snr
    :param category: string that specifies the type of results key. e.g. "energy_information" or "mean_power_information"
    :param value: the value that will be added to the information for the category
    :param result: the current data that has already been saved

    :raises: RuntimeError if the category with board_snr is already set
    :return: dictionary with updated information
    """ 
    
    # board snr should be unique id that serves as key for the information
    id = board_snr

    # get previous category information if exists
    information = {}
    if category in results:

        # get old information of category
        information = results[category]

        # inference information of specified board should not already be contained beforehand
        if id in information:
            raise RuntimeError(f"result.json already contains {category} of the specified board {board_snr}")

    # append new information for board with id
    information[id] = value

    # return the updated information
    return information 


def save_to_dir(board, save_dir, measured_value, tensor_arena_size):
    """ 
    save_to_dir saves the measured values of the serial connection to the result.json under the key inference_information in the save_dir directory. 
    structure of "inference_information" is expected to be this:
    "inference_information" : {
                "board_1" : [board_1_information],
                "board_2: : [board_2_information],
                ....
    }

    :param board: dict containing information about board {"model": <board_model>, "snr": <snr_as_string>}
    :param save_dir: expects a directory path that already contains a result.json containing key vale pairs.
    :param measured_values: a list containing all the measured information.
    :param tensor_arena_size: int containing the size of the tensor arena on the mcu.


    :return: None. Writes the serial input of the board to result.json.

    :raises: RunTimeError if the information of the board is already contained in result.json
    """ 
    # read the values of results.json
    with open(save_dir + '/results.json') as f:
        results = json.loads(f.read())

    # board snr should be unique id that serves as key for the information
    id = board["snr"]

    # append all inference_information (old and new) to the key
    results["inference_information"] = set_result_value_for_board(id,"inference_information", measured_value, results)

    # append all tensorsize_information (old and new) to the key
    results["tensorsize_information"] = set_result_value_for_board(id,"tensorsize_information", tensor_arena_size, results)

    # save it to output
    with open(save_dir + '/results.json', 'w') as f:
        json.dump(results, f, indent=2)


def read_inference_time(board, save_dir = None):
    """ 
    read_inference_time starts a serial connection to the specific board and fetches the inference time and saves it optionally to save_dir else prints it to console.

    :param board: dict containing information about board {"model": <board_model>, "snr": <snr_as_string>}
    :param save_dir (optional): if provided, it expects a directory path that already contains a result.json containing key vale pairs.

    :return: None. Writes the serial input of the board to json or prints it to console.
    """ 

    # getting port for the specific board
    port = None
    try:
        port = get_board_port(board, timeout_in_ms=15000)
    except RuntimeError as e:
        raise NotImplementedError("add proper handling when the board can not be found")

    measured_value = None
    tensor_arena_size = 0
    if port:
        max_iterations_counter = 0
        max_iterations = 28

        received_ready_for_inference = False

        # connecting to port
        with serial.Serial(port, 115200, timeout=0) as ser:
            while measured_value is None:
                time.sleep(0.5)
                try:
                    if received_ready_for_inference:
                        # send 's' to start the inference
                        ser.write(b's')

                    # increase stop criterion counter first before reading
                    if max_iterations_counter > max_iterations:
                        measured_value = "Max iterations reached" + "; Ready received" if received_ready_for_inference else "Max iterations reached" + "; Ready not received"
                        print(measured_value)
                        continue  # this avoids the rare case that at max_iterations reached and it reads one value, thus appending two measured values. maybe rethink the whole structure
                    else:
                        max_iterations_counter = max_iterations_counter + 1
                        if max_iterations_counter % 5 == 0:
                            print(f"max iterations counter: {max_iterations_counter} of {max_iterations}")

                    line = ser.readline()
                    if line != b'':
                        if "Ready" in str(line):
                            print("ready for inference")
                            received_ready_for_inference = True
                        elif "Start" in str(line):
                            print("start inference")
                        elif "AllocateTensor" in str(line) or "failed" in str(line) or "error" in str(line) or "exit" in str(line):
                            print(str(line))
                            measured_value = str(line)
                        elif "InfTime" in str(line):
                            print("inftime")
                            number = re.findall(r'\d+', str(line))[0]
                            inf_time = int(number)
                            print("read inf time:" + str(inf_time))
                            measured_value = inf_time
                        elif "tensorarena" in str(line):
                            tensor_arena_size = int(re.findall(r'\d+', str(line))[0])
                            print(f"tensorarena size captured {tensor_arena_size}")
                        else:
                            print(str(line))

                except Exception as e:
                    print(str(e))

            start = time.time()
            
    else:
         measured_value = "Could not find the port of board. Please connect board."

    end = time.time()
    print("elapsed time: " + str(end - start))

    # save output to directory or print it
    if save_dir is not None:
        save_to_dir(board, save_dir, measured_value, tensor_arena_size)
    else:
        print(measured_value)


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
        "model": args.board_model,
        "snr": args.board_snr
    }
    read_inference_time(board, args.save_dir) 
    time.sleep(2)
