import subprocess
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization


def flash_tflite_model(model_path):

    # load TFLite model
    #model = tf.keras.models.load_model(model_path, custom_objects={"InstanceNormalization": InstanceNormalization})
    tflite_model = tf.lite.Interpreter(model_path=model_path)

    # convert model to TF Lite


    # copy C-array into TFLite projects "model.cpp" file

    # compile airway_tflite project and flash to MCU
    subprocess.run('west build -b nrf52840dk_nrf52840 airway_tflite/', shell=True)
    subprocess.run('west flash', shell=True)


if __name__ == "__main__":
    import sys
    path_to_model = sys.argv[1]
    flash_tflite_model(path_to_model)
