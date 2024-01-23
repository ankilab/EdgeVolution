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


def translate(chromosome: list, input_shape: tuple, num_classes: int, top_activation: str, sample_rate: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    for gene in chromosome:
        if gene['layer'] == 'STFT_2D' and len(model.layers) > 0:
            # Need to check if input_shape is smaller than n_fft, since one output dimension would be 0 then
            # If smaller, use input_shape as n_fft

            previous_layer = model.layers[-1]
            input_shape = previous_layer.input_shape  # shape: (None, shape[0], shape[1])

            if input_shape[1] < gene['n_fft']:
                gene['n_fft'] = input_shape[1]
        elif gene['layer'] == 'FB_2D': 
            # need an extra parameters when applying filterbank

            # sample rate from config.yaml
            gene['sample_rate'] = sample_rate

            # find STFT layer and get its n_fft parameter as it is needed for the filterbank layer
            stft_layer = [x for x in chromosome if x['layer'] == 'STFT_2D'][0]
            gene['n_fft'] = stft_layer['n_fft']

        # make a copy of the gene, since we need to remove the layer key
        gene_copy = copy.deepcopy(gene)
        gene_copy.pop('layer', None)

        # eval the layer name to get the layer
        tf_layer = eval(gene_copy['f_name'])
        if len(gene_copy.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene_copy[x] for x in gene_copy if x not in 'f_name'})))

        # add layer to the model
        model.add(tf_layer)

    # last layer is always classification layer
    model.add(Dense(num_classes, activation=top_activation))
    return model
