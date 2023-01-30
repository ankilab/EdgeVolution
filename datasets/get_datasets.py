import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import signal
import itertools
import os
import pandas as pd
import re
from pathlib import Path

def get_datasets(dataset, samples=6_000, classes_filter=None):
    def _predicate(x, label, allowed_labels=classes_filter):
        allowed_labels = tf.constant(allowed_labels)
        isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
        reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
        return tf.greater(reduced, tf.constant(0.))

    if dataset == "speech_commands":
        """
        [0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes', 
        10: '_silence_', 11: '_unknown_']
        """
        ds_train = tfds.load("speech_commands", data_dir='datasets/', split='train', as_supervised=True, download=True)
        ds_val = tfds.load("speech_commands", data_dir='datasets/', split='validation', as_supervised=True, download=True)
        ds_test = tfds.load("speech_commands", data_dir='datasets/', split='test', as_supervised=True, download=True)

        nb_classes = 12

        def _prepare_dataset(ds, normalize=True):
            if classes_filter is not None:
                ds = ds.filter(_predicate)
                ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
            else:
                ds = ds.map(lambda x, y: (tf.py_function(_resample_func, [x, samples], Tout=tf.float32), tf.one_hot(y, nb_classes)))
            if normalize:
                ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
            return ds

        ds_train = _prepare_dataset(ds_train)
        ds_val = _prepare_dataset(ds_val)
        ds_test = _prepare_dataset(ds_test)

        return ds_train, ds_val, ds_test
    elif dataset == "motion_sense":
        path = Path("datasets/motion_sense_accelerometer")
        classes = {'dws': 0, 'ups': 1, 'sit': 2, 'std': 3, 'wlk': 4, 'jog': 5}
        train_idxs, val_idxs, test_idxs = np.arange(0, 19), np.arange(19, 22), np.arange(22, 25)

        def _get_ds(data, labels, samples=4_000):
            ds = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(data, labels),
                                                output_types=(tf.float64, tf.int32),
                                                output_shapes=((samples, 1), ()))
            if classes_filter is not None:
                ds = ds.map(lambda x, y: (x, tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
            else:
                ds = ds.map(lambda x, y: (x, tf.one_hot(y, len(classes))))
            return ds

        train_data, val_data, test_data = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for folder in os.listdir(path):
            class_path = path / folder
            for f in os.listdir(class_path):
                df = pd.read_csv(class_path / f)

                x, y, z = np.asarray(df['x']), np.asarray(df['y']), np.asarray(df['z'])
                magnitude = list(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
                magnitude = _resample_func(magnitude, samples)
                label = classes[folder[0:3]]

                idx = int(re.findall(r'\d+', f)[0])
                if idx in train_idxs:
                    train_data.append(magnitude)
                    train_labels.append(label)
                elif idx in val_idxs:
                    val_data.append(magnitude)
                    val_labels.append(label)
                else:
                    test_data.append(magnitude)
                    test_labels.append(label)

        ds_train = _get_ds(np.asarray(train_data), np.asarray(train_labels), samples)
        ds_val = _get_ds(np.asarray(val_data), np.asarray(val_labels), samples)
        ds_test = _get_ds(np.asarray(test_data), np.asarray(test_labels), samples)

        return ds_train, ds_val, ds_test

    else:
        raise ValueError(f"Given dataset ({dataset}) is not available.")


def _resample_func(x, samples):
    x = signal.resample(x, samples, axis=0)
    return x[..., np.newaxis]


def _normalize(data, label):
    data /= tf.math.reduce_max(tf.abs(data), axis=0)
    return data, label
