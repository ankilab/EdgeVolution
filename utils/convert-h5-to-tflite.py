import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# TODO: load 200 mel-spectrograms
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 64, 64, 1)
      yield [data.astype(np.float32)]

#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("src/airway_model.hdf5")
model_hdf5 = load_model(r"airway_model.hdf5")
converter = tf.lite.TFLiteConverter.from_keras_model(model_hdf5)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8  

tflite_model = converter.convert()

with open("airway_model.tflite", "wb") as fp:
    fp.write(tflite_model)