"""
Script to download CIFAR-10 dataset
"""

import tensorflow_datasets as tfds


ds_train, ds_val, ds_test = tfds.load('cifar10', data_dir="../data/.", split=['train[:90%]', 'train[90%:]', 'test'],
                                                    as_supervised=True, download=True, with_info=False)