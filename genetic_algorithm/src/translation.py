import copy

import tensorflow as tf
# .layers import Conv2D

from tensorflow.keras.layers import Conv2D, Dense, DepthwiseConv2D, Dense,  GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout, Conv1D,  \
    GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D

from tensorflow.keras.layers import BatchNormalization,DepthwiseConv1D

from ast import literal_eval


def translate(chromosome: list, input_shape: tuple, nb_classes: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Flatten())
    
    # for gene in chromosome:
    #     gene = copy.deepcopy(gene)

    #     gene.pop('layer', None)
    #     tf_layer = eval(gene['f_name'])
    #     if len(gene.keys()) > 1:
    #         tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
    #     # add layer to the model

    #     model.add(tf_layer)

    # last layer is always classification layer
    model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
    return model
