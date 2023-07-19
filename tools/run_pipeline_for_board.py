import time
import os
import shutil
import argparse
import subprocess
from subprocess import Popen
from measure_power_consumption import init_ppk2, stop_measuring
import json


def run_pipeline_for_board(save_dir,board_snr, board_model, ppk_serial, power_measurement_nb_samples_average):
    """ 
    run_pipeline_for_board runs the pipeline (optionally measure power, flashing, infering, measuring inference time) for a specific board.
    
    :param save_dir: directory where to save the csv file, read result.json and optionally save error log text file
    :param board_snr: snr of board connected to ppk to map measured data to a specific board
    :param board_model: model of board connected to ppk to map measured data to a specific board
    :param ppk_serial: serial number of the power profiler that is connected to the board to be measured or "None" if no ppk connected.
    :param power_measurement_nb_samples_average: window size of average sampling 
    
    :return: None
    """

    # init paths
    root_src_folder = './tflite/airway_tflite'
    source_folder_name = f'{board_model}_{board_snr}'
    src_folder = f'./tflite/{source_folder_name}'
    tflite_path = "../" + save_dir + "/models/model_tflite_untrained.tflite"
    board_results_path = save_dir + f'/results_{board_snr}.json'
    cpp_path = f'.{src_folder}/src/model.cpp'
    flasher_path = './tools/flash_tflite_model.sh'


    # copy root src folder for board if doesnt exist yet
    if not os.path.exists(src_folder):
        shutil.copytree(root_src_folder, src_folder)


    with open(board_results_path, 'w') as f:
        json.dump({}, f, indent=2)

    # start measuring
    ppk2 = init_ppk2(ppk_serial)
    time.sleep(1)
    subprocess.call(['bash', '-i',flasher_path, tflite_path, cpp_path, board_model, board_snr, source_folder_name])
    time.sleep(1)

    # if no ppk connected, measuring the power consumption is not possible
    if ppk2 is not None:
        stop_measuring(ppk2)

        # start measuring energy consumption
        args = ['python tools/measure_power_consumption.py', save_dir,board_results_path, board_snr, ppk_serial, power_measurement_nb_samples_average]
        command = " ".join(args) # joining args separated by space
        proc_energy = Popen(command, shell=True)

    # get inference time from Serial port
    args = ['python tools/measure_inference_time.py', save_dir,board_results_path, board_model, board_snr]
    command = " ".join(args) # joining args separated by space
    proc_inference = Popen(command, shell=True)

    # wait for inference time measurement to finish
    proc_inference.wait()

    # if no ppk connected, measuring the power consumption is not possible
    if ppk2 is not None:
        # wait for energy consumption measurement to finish
        try:
            proc_energy.wait(timeout=30)

            # calculate energy consumption
        except:
            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RunPipeline',
        description='This script runs the pipeline (optionally measure power, flashing, infering, measuring inference time) for a specific board.')

    parser.add_argument('save_dir', nargs='?', default=None, help="directory where to save the csv file, read result.json")
    parser.add_argument('board_snr', nargs='?', default=None, help="snr of board connected to ppk to map measured data to a specific board")
    parser.add_argument('board_model', nargs='?', default=None, help="model of board connected to ppk to map measured data to a specific board")
    parser.add_argument('ppk_serial', nargs='?', default=None, help="serial number of the power profiler that is connected to the board to be measured or 'None' if no ppk connected.")
    parser.add_argument('power_measurement_nb_samples_average', nargs='?', default=200, help="samples of power measurement")

    # parse arguments
    args = parser.parse_args()
    run_pipeline_for_board(args.save_dir,args.board_snr, args.board_model, args.ppk_serial, args.power_measurement_nb_samples_average)

