import copy

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Activation, Flatten, Dropout, Resizing, Conv1D, DepthwiseConv1D, \
    GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
from tensorflow_addons.layers import InstanceNormalization
from ast import literal_eval

from kapre import STFT, Magnitude, MagnitudeToDecibel

from genetic_algorithm.src.genepool_modules.sinc_conv_layer import SincConv1D
from genetic_algorithm.src.genepool_modules.filterbank_layer import get_filterbank_layer


def translate(chromosome: list, input_shape: tuple, nb_classes: int, sample_rate: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    for gene in chromosome:
        gene = copy.deepcopy(gene)

        # need an extra parameters when applying filterbank
        if gene['layer'] == 'FB_2D':
            # sample rate from config.yaml
            gene['sample_rate'] = sample_rate

            # find STFT layer and get its n_fft parameter as it is needed for the filterbank layer
            stft_layer = [x for x in chromosome if x['layer'] == 'STFT_2D'][0]
            gene['n_fft'] = stft_layer['n_fft']

        gene.pop('layer', None)
        tf_layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))

        # add layer to the model
        model.add(tf_layer)

    # last layer is always classification layer
    # model.add(Dense(nb_classes, activation='softmax'))
    model.add(Dense(nb_classes, activation='sigmoid'))
    return model
