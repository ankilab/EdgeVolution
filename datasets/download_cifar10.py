"""
Script to download CIFAR-10 dataset
"""

from load_data import get_datasets
import tensorflow_datasets as tfds


ds_train, ds_val, ds_test = tfds.load('cifar10', data_dir=".", split=['train[:90%]', 'train[90%:]', 'test'],
                                                    as_supervised=True, download=True, with_info=False)