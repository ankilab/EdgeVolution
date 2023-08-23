import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

import tensorflow as tf
import numpy as np
from genetic_algorithm.src.translation import translate
from genetic_algorithm.utils.convert_to_tflite import convert_to_tflite
import time
from multiprocessing import Pool
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 

# this is old code that serves for benchmarking
def sequential_convert_to_tflite(model, representative_data=None):
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

def f(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


if __name__ == "__main__":
    print("testing tflite conversions...")

    # sample model to be converted
    chromosome = [
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 3,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 2,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "MP_1D",
        "f_name": "MaxPooling1D",
        "pool_size": 2,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 4,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "AP_1D",
        "f_name": "AveragePooling1D",
        "pool_size": 3,
        "padding": "same"
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 21,
        "kernel_size": 1,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 1,
        "strides": 1,
        "padding": "same"
    },
    {
        "layer": "MP_1D",
        "f_name": "MaxPooling1D",
        "pool_size": 2,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 1,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "BN_1D",
        "f_name": "BatchNormalization()",
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 29,
        "kernel_size": 2,
        "strides": 1,
        "padding": "same"
    },
    {
        "layer": "MP_1D",
        "f_name": "MaxPooling1D",
        "pool_size": 4,
        "padding": "same"
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 10,
        "kernel_size": 3,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "AP_1D",
        "f_name": "AveragePooling1D",
        "pool_size": 4,
        "padding": "same"
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 19,
        "kernel_size": 2,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "AP_1D",
        "f_name": "AveragePooling1D",
        "pool_size": 2,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 2,
        "strides": 1,
        "padding": "same"
    },
    {
        "layer": "AP_1D",
        "f_name": "AveragePooling1D",
        "pool_size": 3,
        "padding": "same"
    },
    {
        "layer": "BN_1D",
        "f_name": "BatchNormalization()"
    },
    {
        "layer": "R_1D",
        "f_name": "ReLU()"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 4,
        "strides": 1,
        "padding": "same"
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 20,
        "kernel_size": 2,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "GMP_1D",
        "f_name": "GlobalMaxPooling1D()"
    },
    {
        "layer": "DO",
        "f_name": "Dropout",
        "rate": 0.30000000000000004
    },
    {
        "layer": "D",
        "f_name": "Dense",
        "units": 112,
        "activation": "relu"
    },
    {
        "layer": "DO",
        "f_name": "Dropout",
        "rate": 0.0
    },
    {
        "layer": "D",
        "f_name": "Dense",
        "units": 88,
        "activation": "relu"
    },
    {
        "layer": "DO",
        "f_name": "Dropout",
        "rate": 0.0
    },
    {
        "layer": "D",
        "f_name": "Dense",
        "units": 32,
        "activation": "relu"
    }
    ]
    input_shape = (6_000, 1)
    model = translate(chromosome,input_shape,12,16000)
    x = 10
    models = [model] * x

    # run sequential for x times
    start = time.time()
    for model in models:
        tflite_model = sequential_convert_to_tflite(model)
    sequential_elapsed_time = time.time() - start


    print("=========================================")   
    
    
    #run x parallel conversions
    start = time.time()

    pool = ThreadPool()
    outputs = pool.map(sequential_convert_to_tflite,models)
    pool.close()
    pool.join()

    elapsed_time = time.time() - start


    # results
    print(f"seq: {sequential_elapsed_time} vs {elapsed_time}")




