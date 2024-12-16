import tensorflow as tf
import copy

from neural_architecture_search.utils.prepare_stft_and_fb_genes import prepare_stft_and_fb_genes
from neural_architecture_search.src.layer_definitions import instantiate_layer, get_classification_layer


def translate(chromosome: list, input_shape: tuple, num_classes: int, top_activation: str, sample_rate: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    for gene in chromosome:
        # If pre-processing is needed, do it here
        gene = prepare_stft_and_fb_genes(gene, chromosome, model, sample_rate)  # prepare the gene for STFT and FB layers as needed 
        # ..... add more if needed

        # make a copy of the gene, since we need to remove the layer key but we don't want to modify the original gene as we need it later
        gene_copy = copy.deepcopy(gene)
        layer_name = gene_copy.pop('layer', None)

        # Instantiate the layer dynamically
        try:
            layer = instantiate_layer(gene=gene_copy, layer_name=layer_name)
            model.add(layer)
        except ValueError as e:
            raise ValueError(f"Failed to add layer '{layer_name}' to model: {e}")

    # last layer is always classification layer
    classification_layer = get_classification_layer(num_classes=num_classes, top_activation=top_activation)
    model.add(classification_layer)
    return model
