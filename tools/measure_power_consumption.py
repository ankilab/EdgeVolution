import time
from ppk2_api.ppk2_api import PPK2_MP, PPK2_API
import argparse
import csv
import json
import numpy as np
import signal
import serial.tools.list_ports


def get_ppk_port(ppk_serial:str):
    """ 
    get_ppk_port returns the port of the connected ppk.

    :param ppk_serial: serial number of the power profiler that is connected to the board to be measured.

    :return: port device if corresponding ppk is found

    :raises: RuntimeError if there corresponding ppk is not found or multiple with same serial number are connected.  
    """ 
    # store all potential devices in array to ensure the right device is selected
    devices = []
    
    # get ports
    ports = serial.tools.list_ports.comports()

    # find port that contains the ppk serial number
    for port in ports:

        if port.serial_number is not None:
            if ppk_serial in port.serial_number:
                devices.append(port.device)
    
    # return device if single device is found
    if len(devices) == 1:
        return devices[0]
    
    # raise error if no device is found
    if len(devices) == 0:
        raise RuntimeError(f"The ppk with serial number {ppk_serial} does not seem to be connected")
    
    # raise error if more than 1 device is found. means that there are two ppks with same serial number or the first 8 digits are not sufficient to uniquely identify ppk
    if len(devices) > 1:
        raise RuntimeError(f"There is more than one ppk with serial number {ppk_serial} connected")


def init_ppk2(ppk_serial: str, timeout_in_s=10):
    """ 
    init_ppk2 initializes the ppk and starts measuring.

    :param ppk_serial: serial number of the power profiler that is connected to the board to be measured or "None" if no ppk connected.
    :param timeout_in_s: timeout to stop trying after unsuccessful initialization

    :return: ppk2 connection

    :raises: RuntimeError if there corresponding ppk is not found or multiple with same serial number are connected.  
    """ 

    # returns if there is no ppk supposed to be connected
    if ppk_serial == "None":
        return None
    
    # get connected ppk by serial number
    try:
        connected_ppk2_port = get_ppk_port(ppk_serial)

        # init connection
        _ppk2 = PPK2_API(port=connected_ppk2_port); time.sleep(0.1)
        #_ppk2 = PPK2_MP(port=connected_ppk2_port, buffer_max_size_seconds=2, buffer_chunk_seconds=1); time.sleep(0.1)
    except Exception as e:
        print(str(e))

    # init counter
    i = 0    

    # after unsuccessful try, stops timer and waits delay_in_s
    delay_in_s = 2

    # timeout_in_s = max_iterations x delay_in_s 
    max_iterations = int(timeout_in_s/delay_in_s)

    while True:
        try:
            # configure device
            _ppk2.get_modifiers(); time.sleep(0.1)
            _ppk2.use_ampere_meter(); time.sleep(0.1)  # set ampere meter mode
            _ppk2.set_source_voltage(3300); time.sleep(0.1) # set source voltage
            _ppk2.toggle_DUT_power("ON"); time.sleep(0.1)  # enable DUT power

            # start measuring
            _ppk2.start_measuring(); time.sleep(0.1)

            # breaking if successful
            break

        except:
            print(f"Tried to connect to PPK2 with serial {ppk_serial}. Waiting {delay_in_s} seconds.\n")

            # stop measuring
            try:
                _ppk2.stop_measuring()
            except:
                pass

            # delay after unsuccessful try
            time.sleep(delay_in_s)

            # make sure to not be in an infinity loop if the error won't go away
            i += 1

            # checking stop criterion
            if i == max_iterations:
                print(f"Left power measurement initialization infinity loop unsuccessfully after {timeout_in_s} seconds.")
                break
    return _ppk2


def _timeout_handler(signal_number, current_stack):
    """ 
    _timeout_handler is a timeout_handler for measure_power_nrf. 

    return: None

    raises: Exception
    """
    ppk2.stop_measuring()
    time.sleep(1)
    del ppk2
    time.sleep(3)

    raise Exception(f"end of time; signal number: {signal_number}, current stack: {current_stack}")


def measure_power_nrf(_ppk2: PPK2_API, save_dir: str, board_snr: str, ppk_serial: str, nb_samples_average: int, max_iterations=None):
    """ 
    measure_power_nrf measures the power consumption and saves all values to a .csv file at the given location. 
    
    :param _ppk2: ppk connection
    :param save_dir: directory where to save the csv file, read result.json and optionally save error log text file
    :param board_snr: snr of board connected to ppk to map measured data to a specific board
    :param ppk_serial: serial number of the power profiler that is connected to the board to be measured or "None" if no ppk connected.
    :param nb_samples_average: window size of average sampling 
    :param max_iterations (optional): stopping criterion for sampling
    
    :return: None

    :raises: RuntimeError if no ppk is provided. 
    """

    # init iterations counter
    i = 0

    # add timeout handler
    signal.signal(signal.SIGALRM, _timeout_handler)

    # get paths FIXME: make paths more robust for e.g. Windows 
    power_measurement_file_name = "power_measurements_" + board_snr +".csv"
    csv_path = save_dir + "/" + power_measurement_file_name
    error_log_path = save_dir + "/" + "error_log_" + board_snr + ".txt"
    results_path = save_dir + "/" + "results.json"

    max_iterations = None
    # open csv to write power measurements
    with open(csv_path, "w") as csv_file:

        # write first line
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Power Consumption'])

        nb_tries = 0
        while True:
            try:
                # define a timeout for get_data method in [s] to avoid being stucked in an endless loop
                signal.alarm(5)

                # call get_data and try to receive data from PPK2
                try:
                    read_data = _ppk2.get_data()
                except Exception as e:
                    nb_tries += 1
                    print(f"Error when calling get_data: {str(e)}")
                    with open(error_log_path, 'a') as file:
                        file.write(f"10010: Error when calling get_data: {str(e)} \n")
                    if nb_tries < 50:
                        continue
                    else:
                        with open(error_log_path, 'a') as file:
                            file.write(f"10010: Tried to call get_data 100 times, but failed. Stopping power measurement. \n")
                        break

                # if data is read
                if read_data != b'':

                    # get samples 
                    samples, _ = _ppk2.get_samples(read_data)

                    # write samples to csv file
                    for ii in range(0, len(samples)):
                        writer.writerow([samples[ii]])
            except:
                print("Get data is not working")

                continue

            i = i + 1

            # check stopping criterion if exists
            if max_iterations is not None:
                if i > max_iterations:
                    print("max iterations reached. breaking")
                    break
            else:

                # check if inference was already captured to stop measuring power
                with open(results_path) as f:
                    try:
                        # load results json
                        results = json.loads(f.read())

                        # check if inference information exists
                        if "inference_information" in results:

                            # check if board snr inference has already been measured
                            if board_snr in results["inference_information"]:
                                # this means inference time of this board was measured, so we want this process to end;
                                # 50 more iterations will be measured until we leave the while loop
                                print("read the inference information, now iterating 20 times more")
                                i = 0
                                max_iterations = 20
                    except Exception as e:
                        print(str(e))
                        with open(error_log_path, 'a') as file:
                            file.write(f"10020: Fatal error when loading results.json in power measurements: {str(e)} \n")
            time.sleep(0.1)


def stop_measuring(_ppk2: PPK2_API, timeout_in_s=2):
    """ 
    stop_measuring stops measuring the ppk trying it for timeout_in_s.

    :param _ppk2: ppk connection
    :param timeout_in_s: timeout to stop trying after unsuccessful trials

    :return: ppk2 connection
    """ 

    # init counter
    i = 0    

    # after unsuccessful try, stops timer and waits delay_in_s
    delay_in_s = 0.1

    # timeout_in_s = max_iterations x delay_in_s 
    max_iterations = int(timeout_in_s/delay_in_s)

    # count to max iterations
    while i < max_iterations:
        try:
            # stop measuring
            _ppk2.stop_measuring()

            # breaking if no errors occurred
            break
        except:
            print("Trying to stop measuring with PPK2.\n")

            # increment the counter if not successful
            i += 1
        time.sleep(delay_in_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MeasurePower',
        description='This script measures the power consumption of the microcontroller with provided board_snr connected to the power profiling kit (ppk) with provided serial ppk_serial.')

    parser.add_argument('save_dir', nargs='?', default='./tflite', help="directory where to save the csv file, read result.json and optionally save error log text file")
    parser.add_argument('board_snr', nargs='?', default=None, help="snr of board connected to ppk to map measured data to a specific board")
    parser.add_argument('ppk_serial', nargs='?', default=None, help="serial number of the power profiler that is connected to the board to be measured or 'None' if no ppk connected.")
    parser.add_argument('nb_samples_average', nargs='?', default="50", help="window size of average sampling ")

    # parse arguments
    args = parser.parse_args()

    # initializing connection and starting to measure
    ppk2 = init_ppk2(args.ppk_serial)
    time.sleep(3)

    # if ppk_serial is "None", no connection is expected, thus ppk2 is also None.
    if ppk2 is not None:
        # writing measures to csv file
        measure_power_nrf(ppk2, args.save_dir, args.board_snr, args.ppk_serial, int(args.nb_samples_average), None)

        # stop measuring
        stop_measuring(ppk2)

        time.sleep(1)
        del ppk2
        time.sleep(3)
