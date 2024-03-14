import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import signal
import glob


class SpeechCommandsDataloader:
    def __init__(self, params: dict):
        self.samples = params['input_shape'][0]
        self.classes_filter = params['classes_filter']  # containing the allowed labels

        self.nb_classes = 12  # number of classes in the complete dataset

    @staticmethod
    def _resample_func(x, samples):
        """
        Resample the given signal to the given number of samples.
        """
        x = signal.resample(x, samples, axis=0)
        return x[..., np.newaxis]

    @staticmethod
    def _normalize(data, label):
        """
        Normalize the given data.
        """
        data /= tf.math.reduce_max(tf.abs(data), axis=0)
        return data, label

    def _predicate(self, x, label):
        """
        Predicate function to filter the dataset based on the allowed labels.
        """
        allowed_labels = tf.constant(self.classes_filter)
        isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
        reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
        return tf.greater(reduced, tf.constant(0.))

    def _prepare_dataset(self, ds, normalize=True):
        if len(self.classes_filter) != 0:
            ds = ds.filter(self._predicate)
            ds = ds.map(lambda x, y: (tf.py_function(self._resample_func, [x, self.samples], Tout=tf.float32),
                                      tf.one_hot(tf.where(tf.equal(y, self.classes_filter))[0],
                                                 len(self.classes_filter))[0]))
        else:
            ds = ds.map(lambda x, y: (
                tf.py_function(self._resample_func, [x, self.samples], Tout=tf.float32), tf.one_hot(y, self.nb_classes)))
        if normalize:
            ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    def load_dataset(self):
        """
        Load the speech commands dataset.
        [0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes',
        10: '_silence_', 11: '_unknown_']

        :return: ds_train, ds_val, ds_test
        """
        # find directory "datasets/" within the project directory
        folder = glob.glob(f"../../**/datasets/", recursive=True)[0]

        # load the dataset
        ds_train = tfds.load("speech_commands", data_dir=folder, split='train', as_supervised=True, download=True)
        ds_val = tfds.load("speech_commands", data_dir=folder, split='validation', as_supervised=True,
                           download=True)
        ds_test = tfds.load("speech_commands", data_dir=folder, split='test', as_supervised=True, download=True)

        ds_train = self._prepare_dataset(ds_train)
        ds_val = self._prepare_dataset(ds_val)
        ds_test = self._prepare_dataset(ds_test)

        return ds_train, ds_val, ds_test
