"""
Script to download Speech Commands dataset
"""

from load_data import get_datasets
import tensorflow_datasets as tfds

ds_train = tfds.load("speech_commands", data_dir=".", split='train', as_supervised=True, download=True, shuffle_files=True)
ds_val = tfds.load("speech_commands", data_dir=".", split='validation', as_supervised=True,
                    download=True)
ds_test = tfds.load("speech_commands", data_dir=".", split='test', as_supervised=True, download=True)
