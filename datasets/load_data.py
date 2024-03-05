import tensorflow as tf
from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy import signal
import itertools


from datasets.src.dataloader_speech_commands import SpeechCommandsDataloader
from datasets.src.dataloader_spoken import SpokenDataLoader
from datasets.src.dataloader_corscience import CorscienceDataLoader


def get_datasets(dataset: str, params: dict):
    """
    Get the train, validation, and test datasets for the specified dataset.

    Args:
        dataset (str): The name of the dataset.
        params (dict): Additional parameters for the dataloader.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.

    Raises:
        ValueError: If the given dataset is not available.
        NotImplementedError: If the specified dataset is not implemented yet.
    """
    if dataset == "speech_commands":
        dataloader_speech_commands = SpeechCommandsDataloader(params)
        ds_train, ds_val, ds_test = dataloader_speech_commands.load_dataset()
        return ds_train, ds_val, ds_test
    
    elif dataset == "spoken":
        dataloader_spoken = SpokenDataLoader("../github_repos/spokeN-100/")
        data = dataloader_spoken.load_waveform(label_type="number")
        ds_train, ds_val, ds_test = data["train"], data["val"], data["test"]
        return ds_train, ds_val, ds_test

    elif dataset == "motion_sense":
        raise NotImplementedError("MotionSense dataset is not implemented yet.")

    elif dataset == "corscience":
        # TODO: remove batch_size
        dataloader_corscience = CorscienceDataLoader("datasets/EKG Daten/DataSegmented/", batch_size=32)
        ds_train, ds_val, ds_test = dataloader_corscience.load_dataset()
        return ds_train, ds_val, ds_test
    
    else:
        raise ValueError(f"Given dataset ({dataset}) is not available.")


# def _resample_func(x, samples):
#     x = signal.resample(x, samples, axis=0)
#     return x[..., np.newaxis]


# def _normalize(data, label):
#     data /= tf.math.reduce_max(tf.abs(data), axis=0)
#     return data, label
