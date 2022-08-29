import copy

import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, GlobalAveragePooling2D, \
    MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ReLU, Flatten, Dropout
from tensorflow_addons.layers import InstanceNormalization
from ast import literal_eval


def translate(chromosome: list, input_shape, nb_classes) -> tensorflow.keras.Model:
    input_layer = Input(input_shape)
    x = input_layer
    for gene in chromosome:
        gene = copy.deepcopy(gene)
        gene.pop('layer', None)
        tf_layer = eval(gene['f_name'])
        if len(gene.keys()) > 1:
            tf_layer = tf_layer(**literal_eval(str({x: gene[x] for x in gene if x not in 'f_name'})))
        # add layer to the model
        try:
            x = tf_layer(x)
        except:
            raise ImportError("Error using a tf-layer specified in the gene_pool.txt. "
                              "Please manually add the import of the following layer in file 'translation.py': "
                              f"{tf_layer}")
    # last layer is always classification layer
    output = Dense(nb_classes, activation='softmax')(x)
    return Model(input_layer, output)
