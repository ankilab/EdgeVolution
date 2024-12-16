import tensorflow as tf


class NormLayer(tf.keras.layers.Layer):
    def init(self):
        super(NormLayer, self).init()

    def build(self, input_shape):
        pass

    def call(self, X):
        X_new = X
        X_std = (X_new - tf.reduce_min(X_new)) / (tf.reduce_max(X_new) - tf.reduce_min(X_new))
        X_scaled = X_std * (1.0 - 0.0) + 0.0
        return X_scaled


def get_norm_layer():
    return NormLayer()
