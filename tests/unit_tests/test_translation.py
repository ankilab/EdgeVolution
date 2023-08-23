from genetic_algorithm.src.translation import translate


def test_translate_length():

    chromosome = [
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 3,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "C_1D",
        "f_name": "Conv1D",
        "filters": 8,
        "kernel_size": 2,
        "strides": 1,
        "padding": "same"
    },
    {
        "layer": "DC_1D",
        "f_name": "DepthwiseConv1D",
        "kernel_size": 2,
        "strides": 2,
        "padding": "same"
    },
    {
        "layer": "MP_1D",
        "f_name": "MaxPooling1D",
        "pool_size": 2,
        "padding": "same"
    }]
    input_shape = (2,3)
    nb_classes = 5
    sample_rate = 16_000
    model = translate(chromosome,input_shape,nb_classes,sample_rate)

    # +1 Dense softmax layer
    expected_layer_length = len(chromosome) + 1

    assert( expected_layer_length== len(model.layers))


def test_layer_order():
    """
    testing if order of chromosome is correct with list
    """
    return

def test_layer_kwargs():
    """
    testing if layer is initialized with optional kwargs
    """
    return

def test_mel_used():
    """
    testing if mel used has expected bahaviour
    """
    return

