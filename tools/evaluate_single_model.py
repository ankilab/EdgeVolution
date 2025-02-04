import tensorflow as tf
import argparse
import time
import subprocess
from subprocess import Popen
from pathlib import Path
import json
import sys
import os
import yaml
import pandas as pd
import numpy as np
from kapre import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

from measure_power_consumption import init_ppk2

sys.path.append("../")
from neural_architecture_search.utils.save_ram_rom_usage import save_ram_rom_usage
from neural_architecture_search.utils.substitute_tflite_layer import substitute_tflite_layer
from neural_architecture_search.utils.convert_to_tflite import convert_to_tflite

FLASHER_PATH = "flash_tflite_model.sh"
CPP_PATH = "../tflite/evonas_tflite/src/model.cpp"

# Settings for power consumption measurement
POWER_MEASUREMENT_NB_SAMPLES_AVERAGE = 50
POWER_MEASUREMENT_THRESHOLD = 4000  # in uA

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def _get_board_information(board: str):
    boards_fodler = "../conf/boards"
    
    # remove "none" from available boards
    available_boards = [b.split(".")[0] for b in os.listdir(boards_fodler) if b != "none.yaml" and b != "README.md"]
    
    if board not in available_boards:
        raise ValueError(f"Board {board} not found in available boards: {available_boards}")
    else:
        with open(f"{boards_fodler}/{board}.yaml", 'r') as stream:
            board_info = yaml.safe_load(stream)
            board_model = board_info['value'][0]['model']
            board_snr = board_info['value'][0]['snr']
            ppk2_snr = board_info['value'][0]['ppk']
    
    return board_model, board_snr, ppk2_snr

def evaluate_single_model(tflite_path: str, board: str):
    
    board_model, board_snr, ppk2_snr = _get_board_information(board)
    print(f"Board model: {board_model}, Board SNR: {board_snr}, PPK2 SNR: {ppk2_snr}")

#     if ".h5" in tflite_path:
#         input_shape = (8000, 1)  # TODO: make this more generic
#         model = tf.keras.models.load_model(tflite_path, custom_objects={'STFT': STFT,
#                                                          'Magnitude': Magnitude,
#                                                          'ApplyFilterbank': ApplyFilterbank,
#                                                          'MagnitudeToDecibel': MagnitudeToDecibel})
#         tflite_model = substitute_tflite_layer(model, input_shape)
#         tflite_model = convert_to_tflite(tflite_model, np.random.uniform(size=(200, input_shape[0], 1)))

#         tflite_path = tflite_path.replace(".h5", ".tflite")
#         with open(tflite_path, 'wb') as f:
#             f.write(tflite_model)

#     results_path = Path(tflite_path).parent / "results.json"

#     if results_path.exists():
#         # delete file
#         results_path.unlink()

#     with results_path.open("w") as f:
#         d = {}
#         json.dump(d, f)

#     ppk2 = init_ppk2(ppk2_snr)
#     time.sleep(1)  # --> important to wait a bit before flashing the model

#     # flash tflite model on board
#     try:
#         ret_val = subprocess.call(['bash', '-i', FLASHER_PATH, tflite_path, CPP_PATH, board_model, board_snr])
#     except Exception as e:
#         raise e
    
#     if ret_val != 0:
#         raise Exception("Error flashing the model on the board. Ret val: ", ret_val)
# #
#     save_ram_rom_usage(f"../tflite/build-{board_model}", str(results_path))

#     proc_energy = None
#     if ppk2 is not None:
#         del ppk2
#         time.sleep(3)

#         # start measuring energy consumption
#         args = ['python measure_power_consumption.py', str(results_path.parent), board_snr,
#                 ppk2_snr, f'{POWER_MEASUREMENT_NB_SAMPLES_AVERAGE}']
#         command = " ".join(args)  # joining args separated by space
#         proc_energy = Popen(command, shell=True)

#         time.sleep(2)

#     # get inference time from Serial port
#     args = ['python measure_inference_time.py', str(results_path.parent), board_model, board_snr]
#     command = " ".join(args)  # joining args separated by space
#     proc_inference = Popen(command, shell=True)

#     # wait for inference time measurement to finish
#     proc_inference.wait()

#     # if no ppk connected, measuring the power consumption is not possible
#     if proc_energy is not None:
#         # wait for energy consumption measurement to finish
#         proc_energy.wait(timeout=10)

#     # calculate energy consumption
#     _calculate_energy_consumption(board_snr, results_path)


def _calculate_energy_consumption(board_snr: str, results_path: Path):
    """
    calculate_energy_consumption reads the energy measurements from the correct csv, averages it and then integrates it over inference time
    :param board_snr: string containing information the board snr
    :param data_dir: directory path containing result.json and power_measurements_<board_snr>.csv of the board
    :return: None. Writes energy consumption and mean power consumption to results.json
    """

    try:
        # load results from json
        with open(results_path) as f:
            results = json.loads(f.read())
    except FileNotFoundError as e:
        raise NotImplementedError(
            "Not implemented proper handling if result does not exist. "
            "Should actually not be the case and not be ignored")
    except Exception as e:
        raise NotImplementedError("proper error handling")

    try:
        data = pd.read_csv(str(results_path).replace("results.json", f"power_measurements_{board_snr}.csv"))
        # get all power consumption measurements
        values = np.asarray(data["Power Consumption"])

        # omit the first 10k values as they are not stable
        values = values[10000:]
        values_averaged = pd.Series(values).rolling(POWER_MEASUREMENT_NB_SAMPLES_AVERAGE).mean()
        start = np.where(values_averaged > POWER_MEASUREMENT_THRESHOLD)[0][0]
        end = np.where(values_averaged < POWER_MEASUREMENT_THRESHOLD)[0][
            np.where(values_averaged < POWER_MEASUREMENT_THRESHOLD)[0] >
            np.where(values_averaged > POWER_MEASUREMENT_THRESHOLD)[0][0]][0]

        # the value with the highest gradient is
        mean_power_consumption = np.median(values[start:end])  # measured in uA
        mean_power_consumption = mean_power_consumption * (10 ** -6)  # in A
        voltage = 3.3  # in V
        # get inference time from board
        try:
            inf_time = float(results["inference_information"][board_snr])
        except ValueError as e:
            inf_time = 0

        # convert to seconds
        inf_time = inf_time * (10 ** -3)  # in s
        # calculate energy by Energy = Voltage x Current x time
        energy_consumption = voltage * mean_power_consumption * inf_time  # in J
        energy_consumption = energy_consumption * (10 ** 3)  # in mJ

        # save energy consumption to results
        results["energy_information"] = set_result_value_for_board(board_snr, "energy_information",
                                                                   float(energy_consumption), results)
        results["mean_power_information"] = set_result_value_for_board(board_snr, "mean_power_information",
                                                                       float(mean_power_consumption), results)
    except Exception as e:
        results["energy_information"] = set_result_value_for_board(board_snr, "energy", str(e), results)
    # save to results.json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def set_result_value_for_board(board_snr, category, value, results):
    # board snr should be unique id that serves as key for the information
    _id = board_snr

    # get previous category information if exists
    information = {}
    if category in results:
        # get old information of category
        information = results[category]

    # append new information for board with id
    information[_id] = value
    # return the updated information
    return information


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Evaluate Single Model',
        description='This script evaluates (RAM footprint, inference time and power consumption) a single model.'
    )

    parser.add_argument('--model_path', nargs='?', default=None, help='Path to the model (.h5 or .tflite) to be evaluated.')
    parser.add_argument('--board', nargs='?', default=None)
    args = parser.parse_args()

    evaluate_single_model(args.model_path, args.board)
