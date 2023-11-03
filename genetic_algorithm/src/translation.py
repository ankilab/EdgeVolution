import copy

import tensorflow as tf
from tensorflow.python.keras.layers import Dense

# from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
#     MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout, Resizing, Conv1D, DepthwiseConv1D, \
#     GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
from ast import literal_eval


def translate(chromosome: list, input_shape: tuple, nb_classes: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    for gene in chromosome:
        gene = copy.deepcopy(gene)

        gene.pop('layer', None)
        tf_layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
        # add layer to the model

        model.add(tf_layer)

    # last layer is always classification layer
    model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
    return model
