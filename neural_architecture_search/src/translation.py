import tensorflow as tf
import copy

from neural_architecture_search.utils.prepare_stft_and_fb_genes import prepare_stft_and_fb_genes
from neural_architecture_search.src.layer_definitions import instantiate_layer, get_classification_layer


def translate(chromosome: list, input_shape: tuple, num_classes: int, top_activation: str, sample_rate: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    _global_pooling_layers = {'GAP_2D', 'GMP_2D', 'GAP_1D', 'GMP_1D'}

    for gene in chromosome:
        # If pre-processing is needed, do it here
        gene = prepare_stft_and_fb_genes(gene, chromosome, model, sample_rate)  # prepare the gene for STFT and FB layers as needed
        # ..... add more if needed

        # make a copy of the gene, since we need to remove the layer key but we don't want to modify the original gene as we need it later
        gene_copy = copy.deepcopy(gene)
        layer_name = gene_copy.pop('layer', None)

        # Skip global pooling if the tensor is already 2D (e.g. a prior
        # global pooling already flattened it).  This can happen when
        # crossover combines two chromosome tails that each end with a
        # global pooling layer.
        if layer_name in _global_pooling_layers and len(model.output_shape) == 2:
            continue

        # Instantiate the layer dynamically
        try:
            layer = instantiate_layer(gene=gene_copy, layer_name=layer_name)
            model.add(layer)
        except ValueError as e:
            raise ValueError(f"Failed to add layer '{layer_name}' to model: {e}")

    # Ensure the tensor is 2D before classification.  Chromosomes from
    # GenePool always contain a GAP/GMP layer, but those generated via
    # SearchSpaceRegistry (e.g. Ax/BO) may not.
    ndim = len(model.output_shape)
    if ndim == 4:
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif ndim == 3:
        model.add(tf.keras.layers.GlobalAveragePooling1D())

    # last layer is always classification layer
    classification_layer = get_classification_layer(num_classes=num_classes, top_activation=top_activation)
    model.add(classification_layer)
    return model
