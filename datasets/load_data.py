import tensorflow as tf
from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy import signal
import itertools
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.src.dataloader_speech_commands import SpeechCommandsDataloader
from datasets.src.dataloader_cifar10 import Cifar10DataLoader
from datasets.src.dataloader_daliac import DaliacDataLoader
from datasets.src.dataloader_emg_airob import EmgAirobDataLoader


def get_datasets(dataset: str, params: dict = None, return_one_hot: bool = False):
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

    ###########################################################################
    # Speech Commands
    ###########################################################################
    if dataset == "speech_commands":
        dataloader_speech_commands = SpeechCommandsDataloader(params)
        ds_train, ds_val, ds_test = dataloader_speech_commands.load_dataset()

        # class weights are hard-coded to avoid re-calculating them every time
        # they are determined by the following out-commented code
        # class_weights = get_class_weights(ds_train, params["num_classes"])

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

        return ds_train, ds_val, ds_test, class_weights

    ###########################################################################
    # Cifar-10
    ###########################################################################
    elif dataset == "cifar10":
        dataloader_cifar10 = Cifar10DataLoader()
        ds_train, ds_val, ds_test = dataloader_cifar10.load_dataset()
        return ds_train, ds_val, ds_test, None

    ###########################################################################
    # EMG AIROB Lab
    ###########################################################################
    elif dataset == "emg_airob":
        dataloader_emg_airob = EmgAirobDataLoader("emg_airob/merged_EMG_new_raw_data.csv")
        ds_train, ds_val, ds_test = dataloader_emg_airob.load_dataset()
        return ds_train, ds_val, ds_test, None
    
    ###########################################################################
    # DaLiAc
    ###########################################################################
    elif dataset == "daliac":
        dataloader_daliac = DaliacDataLoader(return_one_hot=True)
        ds_train, ds_val, ds_test = dataloader_daliac.load_dataset()
        return ds_train, ds_val, ds_test, None
    else:
        raise ValueError(f"Given dataset ({dataset}) is not available.")
    

def get_class_weights(ds, num_classes):
    """
    Get the class weights for the given dataset.
    """
    class_weights = {}
    total = 0
    for i in range(num_classes):
        filtered = ds.filter(lambda x, y: tf.equal(tf.argmax(y), i))
        class_weights[i] = len(list(filtered.as_numpy_iterator()))
        total += class_weights[i]

    print(class_weights)
    for i in range(num_classes):
        class_weights[i] = total / (num_classes * class_weights[i])
    return class_weights
