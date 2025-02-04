"""
Script to download Speech Commands dataset
"""

import tensorflow_datasets as tfds

ds_train = tfds.load("speech_commands", data_dir="../data/.", split='train', as_supervised=True, download=True, shuffle_files=True)
ds_val = tfds.load("speech_commands", data_dir="../data/.", split='validation', as_supervised=True,
                    download=True)
ds_test = tfds.load("speech_commands", data_dir="../data/.", split='test', as_supervised=True, download=True)
