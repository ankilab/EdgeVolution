import copy

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout, Resizing
from tensorflow_addons.layers import InstanceNormalization
from ast import literal_eval

from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank
from kapre.composed import get_melspectrogram_layer
from genetic_algorithm.utils.norm_layer import get_norm_layer


def translate(chromosome: list, input_shape: tuple, nb_classes: int, sample_rate: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    mel_used = False
    for gene in chromosome:
        gene = copy.deepcopy(gene)
        # need an extra parameter specified in main.py when we have Mel Spectrogram TF-Functional in the beginning
        if gene['layer'] == 'MEL':
            mel_used = True
            gene['sample_rate'] = sample_rate

        gene.pop('layer', None)
        tf_layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
        # add layer to the model
        if mel_used is True:
            for layer in tf_layer.layers:
                model.add(layer)
            mel_used = False
        else:
            model.add(tf_layer)

    # last layer is always classification layer
    model.add(Dense(nb_classes, activation='softmax'))
    return model
