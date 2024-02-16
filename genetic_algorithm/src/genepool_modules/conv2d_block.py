import tensorflow as tf
import numpy as np



def get_conv2d_block(filters, kernel_height, kernel_width, strides, padding, norm_layer, activation):
    """
    Returns a convolutional block with the given parameters.
    """
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width), strides=strides, padding=padding))

    if norm_layer == "BatchNormalization":
        block.add(tf.keras.layers.BatchNormalization())

    if activation != "None":
        block.add(tf.keras.layers.Activation(activation))

    return block

def get_depthwise_conv2d_block(kernel_height, kernel_width, strides, padding, norm_layer, activation):
    """
    Returns a depthwise convolutional block with the given parameters.
    """
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(kernel_height, kernel_width), strides=strides, padding=padding))

    if norm_layer == "BatchNormalization":
        block.add(tf.keras.layers.BatchNormalization())

    if activation != "None":
        block.add(tf.keras.layers.Activation(activation))

    return block

