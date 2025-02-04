from datasets.utils.registry import register_dataset
from datasets.src.base_dataloader import BaseDataLoader

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import signal
import os

@register_dataset("speech_commands")
class SpeechCommandsDataloader(BaseDataLoader):
    def __init__(self, samples: int = 6000, classes_filter: list = []):
        """
        Initialize the speech commands dataset loader.
        :param samples: number of samples
        :param classes_filter: classes to be used (empty list means all classes)
        """
        self.samples = samples
        self.classes_filter = classes_filter

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
    
    def get_class_weights(self, ds, num_classes, recompute=False):
        """
        Get the class weights for the given dataset.
        """
        if recompute:
            class_weights = {}
            total = 0
            for i in range(num_classes):
                filtered = ds.filter(lambda x, y: tf.equal(tf.argmax(y), i))
                class_weights[i] = len(list(filtered.as_numpy_iterator()))
                total += class_weights[i]

            for i in range(num_classes):
                class_weights[i] = total / (num_classes * class_weights[i])
            return class_weights
        else:
            # Ran the above code and saved the class weights here to avoid recomputation every time 
            class_weights = {
                0: 2.273744947883429,
                1: 2.294242326679545,
                2: 2.3463670288662057,
                3: 2.276650692225772,
                4: 2.3992985409652077,
                5: 2.309111039101318,
                6: 2.360356630230761,
                7: 2.2905550198221367,
                8: 2.417203753957485,
                9: 2.207533044196613,
                10: 10.667539920159681,
                11: 0.1317808312066181
            }
            return class_weights

    def load_dataset(self):
        """
        Load the speech commands dataset.
        [0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes',
        10: '_silence_', 11: '_unknown_']

        :return: ds_train, ds_val, ds_test
        """
        # go dynamically back in directory until folder "EdgeVolution" is reached
        folder = os.getcwd()
        while os.path.basename(folder) != "EdgeVolution":
            folder = os.path.dirname(folder)
        
        # go to the datasets folder
        folder = os.path.join(folder, "datasets/data")

        # load the dataset
        ds_train = tfds.load("speech_commands", data_dir=folder, split='train', as_supervised=True, download=True, shuffle_files=True)
        ds_val = tfds.load("speech_commands", data_dir=folder, split='validation', as_supervised=True,
                           download=True)
        ds_test = tfds.load("speech_commands", data_dir=folder, split='test', as_supervised=True, download=True)

        ds_train = self._prepare_dataset(ds_train)
        ds_val = self._prepare_dataset(ds_val)
        ds_test = self._prepare_dataset(ds_test)
  
        class_weights = self.get_class_weights(ds_train, self.nb_classes, recompute=False)

        return ds_train, ds_val, ds_test, class_weights
