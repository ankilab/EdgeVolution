import subprocess
import threading
import time

BOARDS = [
    {"model" : "nrf52840dk_nrf52840", "snr": "1050242564", "ppk": "FEA55411"},
    {"model" : "nrf5340dk_nrf5340_cpuapp", "snr": "1050006605", "ppk": "None"},
]


def run_script(flasher_path, tflite_path, cpp_path, board_model, board_snr, source_folder_name):
    subprocess.call(['bash', '-i',flasher_path, tflite_path, cpp_path, board_model, board_snr, source_folder_name])

threads = []

for board in BOARDS:
    board_model = board["model"]
    board_snr = board["snr"]
    save_dir = "./save_dir"
    root_src_folder = './tflite/airway_tflite'
    source_folder_name = f'{board_model}_{board_snr}'
    src_folder = f'./tflite/{source_folder_name}'
    tflite_path = "../" + save_dir + "/model_tflite_untrained.tflite"
    board_results_path = save_dir + f'/results_{board_snr}.json'
    cpp_path = f'.{src_folder}/src/model.cpp'
    flasher_path = './save_dir/just_build.sh'

    # Create two thread objects, one for each script
    thread = threading.Thread(target=run_script, args=(flasher_path, tflite_path, cpp_path, board_model, board_snr, source_folder_name))
    threads.append(thread)

# Start both threads
for thread in threads:
    thread.start()
    time.sleep(1)

# Wait for both threads to finish
for thread in threads:
    thread.join()

print("Both scripts have finished executing.")