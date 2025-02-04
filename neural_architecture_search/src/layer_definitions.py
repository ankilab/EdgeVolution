# layer_definitions.py
from ast import literal_eval

# Standard TensorFlow layers
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D,
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Activation, Flatten, Dropout, Resizing, 
    Conv1D, DepthwiseConv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
)

# TensorFlow Addons layers
from tensorflow_addons.layers import InstanceNormalization

# Kapre layers
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank

# Custom layers
from neural_architecture_search.src.search_space_modules.sinc_conv_layer import SincConv1D
from neural_architecture_search.src.search_space_modules.filterbank_layer import get_filterbank_layer
from neural_architecture_search.src.search_space_modules.conv2d_block import get_conv2d_block, get_depthwise_conv2d_block

# Define custom objects for model loading
CUSTOM_OBJECTS = {
    'SincConv1D': SincConv1D,
    'STFT': STFT,
    'Magnitude': Magnitude,
    'ApplyFilterbank': ApplyFilterbank,
    'MagnitudeToDecibel': MagnitudeToDecibel,
    'InstanceNormalization': InstanceNormalization
}

# Helper function for layer instantiation
def instantiate_layer(gene: dict, layer_name: str):
    """
    Dynamically creates a layer instance based on the layer name and keyword arguments.

    :param layer_name: Name of the layer (string)
    :param kwargs: Parameters to pass to the layer constructor
    :return: Instantiated layer
    """
    try:
        # eval the layer name to get the layer
        layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            # if there are additional parameters, pass them to the layer
            layer = layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
        return layer
    except KeyError:
        raise ValueError(f"Layer '{layer_name}' is not defined in layer_definitions.py")


def get_classification_layer(num_classes: int, top_activation: str):
    """
    Returns a classification layer based on the number of classes and the top activation function.

    :param num_classes: Number of classes (int)
    :param top_activation: Activation function for the top layer (string)
    :return: Classification layer
    """
    if num_classes > 2:
        return Dense(num_classes, activation=top_activation)
    else:
        return Dense(1, activation='sigmoid')
    
def load_model(model_path, weights_path=None):
    """
    Load a model from a file.

    :param model_path: Path to the model file
    :param weights_path: Path to the weights file
    :return: Loaded model
    """
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    if weights_path is not None:
        model.load_weights(weights_path)
    return model