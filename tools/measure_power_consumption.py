import time
from ppk2_api.ppk2_api import PPK2_MP
import argparse
import csv
import json
import numpy as np
import signal


def init_ppk2():
    ppk2s_connected = PPK2_MP.list_devices()
    _ppk2 = PPK2_MP(port=ppk2s_connected[0], buffer_seconds=2); time.sleep(0.1)

    i = 0
    while True:
        try:
            _ppk2.get_modifiers(); time.sleep(0.1)
            _ppk2.use_ampere_meter(); time.sleep(0.1)  # set ampere meter mode
            _ppk2.set_source_voltage(3300); time.sleep(0.1)
            _ppk2.toggle_DUT_power("ON"); time.sleep(0.1)  # enable DUT power
            _ppk2.start_measuring(); time.sleep(0.1)  # start measuring
            break
        except:
            print("Trying to connect to PPK2.\n")
            _ppk2.stop_measuring()
            time.sleep(1)
            # make sure to not be in an infinity loop if the error won't go away
            i = i + 1
            if i == 10_000:
                print("Left power measurement infinity loop.")
                break
        time.sleep(0.1)

    return _ppk2


def _timeout_handler(signal_number, current_stack):
    raise Exception(f"end of time; signal number: {signal_number}, current stack: {current_stack}")


def measure_power_nrf(_ppk2, save_dir: str, nb_samples_average: int, max_iterations=None):
    """ Measures the power consumption and saves all values to a .csv file at the given location. """
    i = 0
    signal.signal(signal.SIGALRM, _timeout_handler)
    with open(save_dir, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        #writer.writerow(['Timestamp', 'Power Consumption'])
        writer.writerow(['Power Consumption'])

        while True:
            try:
                # define a timeout for get_data method in [s] to avoid being stucked in an endless loop
                signal.alarm(5)

                # call get_data and try to receive data from PPK2
                read_data = _ppk2.get_data()
                if read_data != b'':
                    samples = _ppk2.get_samples(read_data)

                    #t = time.time()
                    for i in range(0, len(samples), nb_samples_average):
                        #writer.writerow([t, np.mean(samples[i:i + nb_samples_average])])
                        writer.writerow([np.mean(samples[i:i + nb_samples_average])])
            except:
                print("Get data is not working")
                _ppk2 = init_ppk2()
                time.sleep(1)
                continue

            i = i + 1

            if max_iterations is not None:
                if i > max_iterations:
                    break
            else:
                with open(save_dir.replace("power_measurements.csv", "results.json")) as f:
                    try:
                        d = json.loads(f.read())
                        if "inference_time" in d.keys():
                            # this means inference time was measured, so we want this process to end;
                            # 50 more iterations will be measured until we leave the while loop
                            i = 0
                            max_iterations = 50
                    except Exception as e:
                        with open(save_dir.replace("power_measurements.csv", "error_log.txt"), 'a') as file:
                            file.write(f"10020: Fatal error when loading results.json in power measurements: {str(e)} \n")
            time.sleep(0.1)


def stop_measuring(_ppk2):
    while True:
        try:
            _ppk2.stop_measuring()
            break
        except:
            print("Trying to stop measuring with PPK2.\n")
            pass
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MeasurePower',
        description='This script measures the power consumption of the NRF52840 microcontroller.')

    parser.add_argument('save_dir', nargs='?', default='../tflite/power_measurements.csv')
    parser.add_argument('nb_samples_average', nargs='?', default="2000")
    args = parser.parse_args()

    ppk2 = init_ppk2()
    measure_power_nrf(ppk2, args.save_dir, int(args.nb_samples_average))
    stop_measuring(ppk2)
