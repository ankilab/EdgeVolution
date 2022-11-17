import tensorflow as tf
import numpy as np


def convert_to_tflite(model, representative_data=None):
    # create TFLite converter object
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    if representative_data is not None:
        def representative_dataset():
            for _ in range(len(representative_data)):
                yield [representative_data.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # converter specifications
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    return tflite_model
