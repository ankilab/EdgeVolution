# layer_definitions.py
"""
Layer definitions and instantiation utilities for Neural Architecture Search.

This module provides:
- Layer instantiation from gene dictionaries
- Custom object registry for model loading
- Integration with LayerRegistry for validation

The module maintains backward compatibility with the existing chromosome format
while supporting the new LayerRegistry-based validation.
"""

# Standard TensorFlow layers
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalMaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Resizing,
    Conv1D,
    DepthwiseConv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalMaxPooling1D,
)

# TensorFlow Addons layers
from tensorflow_addons.layers import InstanceNormalization

# Kapre layers
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank

# Custom layers - importing triggers registration with LayerRegistry
from neural_architecture_search.src.search_space_modules.sinc_conv_layer import (
    SincConv1D,
)
from neural_architecture_search.src.search_space_modules.filterbank_layer import (
    get_filterbank_layer,
)
from neural_architecture_search.src.search_space_modules.conv2d_block import (
    get_conv2d_block,
    get_depthwise_conv2d_block,
)

# Import LayerRegistry for validation support
from neural_architecture_search.src.layer_registry import LayerRegistry

# Define custom objects for model loading
CUSTOM_OBJECTS = {
    "SincConv1D": SincConv1D,
    "STFT": STFT,
    "Magnitude": Magnitude,
    "ApplyFilterbank": ApplyFilterbank,
    "MagnitudeToDecibel": MagnitudeToDecibel,
    "InstanceNormalization": InstanceNormalization,
}

# Known layers for safe lookup (replaces eval()-based fallback)
KNOWN_LAYERS = {
    "Conv2D": Conv2D,
    "DepthwiseConv2D": DepthwiseConv2D,
    "Dense": Dense,
    "BatchNormalization": BatchNormalization,
    "GlobalAveragePooling2D": GlobalAveragePooling2D,
    "MaxPooling2D": MaxPooling2D,
    "AveragePooling2D": AveragePooling2D,
    "GlobalMaxPooling2D": GlobalMaxPooling2D,
    "Activation": Activation,
    "Flatten": Flatten,
    "Dropout": Dropout,
    "Resizing": Resizing,
    "Conv1D": Conv1D,
    "DepthwiseConv1D": DepthwiseConv1D,
    "GlobalAveragePooling1D": GlobalAveragePooling1D,
    "MaxPooling1D": MaxPooling1D,
    "AveragePooling1D": AveragePooling1D,
    "GlobalMaxPooling1D": GlobalMaxPooling1D,
    "InstanceNormalization": InstanceNormalization,
    "STFT": STFT,
    "Magnitude": Magnitude,
    "MagnitudeToDecibel": MagnitudeToDecibel,
    "ApplyFilterbank": ApplyFilterbank,
    "SincConv1D": SincConv1D,
    "get_filterbank_layer": get_filterbank_layer,
    "get_conv2d_block": get_conv2d_block,
    "get_depthwise_conv2d_block": get_depthwise_conv2d_block,
}


def instantiate_layer(gene: dict, layer_name: str):
    """
    Dynamically creates a layer instance based on the layer name and gene parameters.

    This function supports two modes:
    1. Registry mode (preferred): Uses LayerRegistry for lookup and instantiation
    2. Known layers fallback: Uses the KNOWN_LAYERS dict for layers not in the registry

    Args:
        gene: Dictionary containing layer parameters including 'f_name'
        layer_name: Name of the layer (for error messages)

    Returns:
        Instantiated Keras layer

    Raises:
        ValueError: If the layer is not defined
    """
    try:
        f_name = gene["f_name"]

        # Check if we can use the registry (preferred)
        clean_name = f_name.rstrip("()")
        if LayerRegistry.exists(clean_name):
            layer_fn = LayerRegistry.get(clean_name)
        elif clean_name in KNOWN_LAYERS:
            # Fallback to known layers dict (safe alternative to eval)
            layer_fn = KNOWN_LAYERS[clean_name]
        else:
            raise ValueError(
                f"Layer '{layer_name}' uses unknown f_name '{f_name}'. "
                f"Register it with @LayerRegistry.register() or add it to KNOWN_LAYERS."
            )

        # If f_name ends with (), it's a no-arg instantiation
        if f_name.endswith("()"):
            return layer_fn()

        # Otherwise, pass parameters
        params = {k: v for k, v in gene.items() if k not in ("layer", "f_name")}
        if params:
            return layer_fn(**params)
        else:
            return layer_fn()

    except KeyError:
        raise ValueError(f"Layer '{layer_name}' is missing 'f_name' in gene: {gene}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to instantiate layer '{layer_name}' with gene {gene}: {e}"
        )


def get_classification_layer(num_classes: int, top_activation: str):
    """
    Returns a classification layer based on the number of classes and activation.

    Args:
        num_classes: Number of output classes
        top_activation: Activation function for the output layer

    Returns:
        Dense layer configured for classification
    """
    if num_classes > 2:
        return Dense(num_classes, activation=top_activation)
    else:
        return Dense(1, activation="sigmoid")


def load_model(model_path, weights_path=None):
    """
    Load a model from a file.

    Args:
        model_path: Path to the model file
        weights_path: Optional path to weights file

    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def validate_chromosome(chromosome: list) -> bool:
    """
    Validate that all layers in a chromosome are registered.

    Args:
        chromosome: List of gene dictionaries

    Returns:
        True if all layers are valid

    Raises:
        ValueError: If any layer is not registered
    """
    for gene in chromosome:
        f_name = gene.get("f_name", "")
        clean_name = f_name.rstrip("()")

        if not LayerRegistry.exists(clean_name) and clean_name not in KNOWN_LAYERS:
            raise ValueError(
                f"Layer '{gene.get('layer')}' uses unregistered f_name '{f_name}'. "
                f"Available layers: {', '.join(LayerRegistry.list_available()[:10])}..."
            )

    return True
