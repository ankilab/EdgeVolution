import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import signal


def get_datasets(dataset, samples=6_000, classes_filter=None):
    if dataset == "speech_commands":
        """
        [0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes', 
        10: '_silence_', 11: '_unknown_']
        """
        ds_train = tfds.load("speech_commands", data_dir='datasets/', split='train', as_supervised=True, download=True)
        ds_val = tfds.load("speech_commands", data_dir='datasets/', split='validation', as_supervised=True, download=True)
        ds_test = tfds.load("speech_commands", data_dir='datasets/', split='test', as_supervised=True, download=True)

        nb_classes = 12

        def _predicate(x, label, allowed_labels=classes_filter):
            allowed_labels = tf.constant(allowed_labels)
            isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
            reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
            return tf.greater(reduced, tf.constant(0.))

        def _resample_func(x, samples):
            x = signal.resample(x, samples, axis=0)
            return x[..., np.newaxis]

        def _normalize(data, label):
            data /= tf.math.reduce_max(tf.abs(data), axis=0)
            return data, label
                    
        def _prepare_dataset(ds):
            if classes_filter is not None:
                ds = ds.filter(_predicate)
                ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
            else:
                ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(y, nb_classes)))
            ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
            return ds

        ds_train = _prepare_dataset(ds_train)
        ds_val = _prepare_dataset(ds_val)
        ds_test = _prepare_dataset(ds_test)

        return ds_train, ds_val, ds_test

    else:
        raise ValueError(f"Given dataset ({dataset}) is not available.")
