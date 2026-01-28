import tensorflow as tf

from neural_architecture_search.src.layer_registry import LayerRegistry


@LayerRegistry.register(metadata={"source": "custom", "category": "feature_extraction"})
def get_conv2d_block(
    filters, kernel_height, kernel_width, strides, padding, norm_layer, activation
):
    """
    Returns a convolutional block with the given parameters.

    Args:
        filters: Number of output filters
        kernel_height: Height of the convolution kernel
        kernel_width: Width of the convolution kernel
        strides: Stride of the convolution
        padding: Padding mode ('same' or 'valid')
        norm_layer: Normalization layer to use ('BatchNormalization' or 'None')
        activation: Activation function ('relu', 'sigmoid', etc. or 'None')

    Returns:
        A Sequential model containing the conv block layers
    """
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_height, kernel_width),
            strides=strides,
            padding=padding,
        )
    )

    if norm_layer == "BatchNormalization":
        block.add(tf.keras.layers.BatchNormalization())

    if activation != "None":
        block.add(tf.keras.layers.Activation(activation))

    return block


@LayerRegistry.register(metadata={"source": "custom", "category": "feature_extraction"})
def get_depthwise_conv2d_block(
    kernel_height, kernel_width, strides, padding, norm_layer, activation
):
    """
    Returns a depthwise convolutional block with the given parameters.

    Args:
        kernel_height: Height of the convolution kernel
        kernel_width: Width of the convolution kernel
        strides: Stride of the convolution
        padding: Padding mode ('same' or 'valid')
        norm_layer: Normalization layer to use ('BatchNormalization' or 'None')
        activation: Activation function ('relu', 'sigmoid', etc. or 'None')

    Returns:
        A Sequential model containing the depthwise conv block layers
    """
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_height, kernel_width), strides=strides, padding=padding
        )
    )

    if norm_layer == "BatchNormalization":
        block.add(tf.keras.layers.BatchNormalization())

    if activation != "None":
        block.add(tf.keras.layers.Activation(activation))

    return block
