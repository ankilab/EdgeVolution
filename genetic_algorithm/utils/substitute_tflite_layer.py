import tensorflow as tf
from kapre import STFT, Magnitude, STFTTflite, MagnitudeTflite


def substitute_tflite_layer(model, input_shape):
    """ Preprocessing layers are critical on MCUs since some TF ops (e.g. RANGE) are not implemented for
    TFLite-micro. Therefore, these layers need to be replaced with special implementations to be able to deploy the
    model. """
    model_with_tflite_layers = tf.keras.Sequential()
    model_with_tflite_layers.add(tf.keras.Input(shape=input_shape))

    for layer in model.layers:
        if type(layer) == STFT:
            model_with_tflite_layers.add(
                STFTTflite(input_shape=layer.input_shape[1::], n_fft=layer.n_fft, hop_length=layer.hop_length,
                           input_data_format='channels_last', output_data_format='channels_last'))
        elif type(layer) == Magnitude:
            model_with_tflite_layers.add(MagnitudeTflite())
        else:
            model_with_tflite_layers.add(layer)

    return model_with_tflite_layers
