import tensorflow as tf
import numpy as np

from neural_architecture_search.src.layer_registry import LayerRegistry


def sinc(x):
    return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))


def generate_sinc_kernel(kernel_size, cutoff):
    # Generate an array of equally spaced points from -cutoff to cutoff
    t = np.linspace(-cutoff, cutoff, kernel_size)

    # Calculate the Sinc values for each point
    kernel = sinc(t)

    # Normalize the kernel (optional but recommended)
    kernel /= np.sum(kernel, dtype=np.float32)

    return kernel


@LayerRegistry.register(metadata={"source": "custom", "category": "preprocessing"})
class SincConv1D(tf.keras.layers.Layer):
    """
    SincConv1D layer as described in the paper:
    "Speaker Recognition from raw waveform with SincNet" by Mirco Ravanelli, Yoshua Bengio
    https://arxiv.org/pdf/1808.00158.pdf

    Example usage:
    model = tf.keras.Sequential([
        SincConv1D(num_filters=16, kernel_size=16, cutoff=4),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    """

    def __init__(self, num_filters, kernel_size, cutoff, **kwargs):
        """
        :param num_filters: number of filters
        :param kernel_size: size of the kernel
        :param cutoff: cutoff frequency of the filter
        :param kwargs: other arguments for tf.keras.layers.Layer
        """
        super(SincConv1D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.cutoff = cutoff

    def build(self, input_shape):
        # Generate the Sinc kernel
        sinc_kernel = generate_sinc_kernel(self.kernel_size, self.cutoff)
        sinc_kernel = np.tile(sinc_kernel, (self.num_filters, 1))
        sinc_kernel = np.expand_dims(sinc_kernel, axis=-1)

        # Create a trainable weight variable for the Sinc kernel
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, 1, self.num_filters),
            initializer=tf.keras.initializers.Constant(sinc_kernel),
            trainable=True,
        )

    def call(self, inputs):
        # Apply 1D convolution with the Sinc kernel
        outputs = tf.nn.conv1d(inputs, self.kernel, stride=1, padding="SAME")
        return outputs

    def get_config(self):
        config = super(SincConv1D, self).get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "cutoff": self.cutoff,
            }
        )
        return config
