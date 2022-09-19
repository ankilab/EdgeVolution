import copy

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout, Reshape, Resizing
from kapre.composed import get_melspectrogram_layer
from tensorflow_addons.layers import InstanceNormalization
from ast import literal_eval

from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank
from utils.norm_layer import get_norm_layer


def translate(chromosome: list, input_shape, nb_classes) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    # add preprocessing layers
    # melspectrogram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, win_length=64, hop_length=64,
    #                                           return_decibel=True,
    #                                           n_mels=80, input_data_format='channels_last',
    #                                           output_data_format='channels_last', mel_htk=True)
    #
    # for layer in melspectrogram.layers:
    #     model.add(layer)


    #model.add(get_norm_layer())
    #model.add(tf.keras.layers.Resizing(64, 64))

    for gene in chromosome:
        gene = copy.deepcopy(gene)
        gene.pop('layer', None)
        tf_layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
        # add layer to the model
        model.add(tf_layer)

    # last layer is always classification layer
    model.add(Dense(nb_classes, activation='softmax'))
    return model
