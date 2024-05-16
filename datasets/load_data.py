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
from datasets.src.dataloader_spoken import SpokenDataLoader
from datasets.src.dataloader_corscience import CorscienceDataLoader
from datasets.src.dataloader_daliac import DaliacDataLoader
from datasets.src.dataloader_toy_admos import ToyAdmosDataloader
from datasets.src.dataloader_airway import AIrwayDataLoader

 
def get_datasets(dataset: str, path:str = None, params: dict = None, return_one_hot: bool = False):
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
        # they aredetermined by the following out-commented code
        #class_weights = get_class_weights(ds_train, params["num_classes"])

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

        # def lr_schedule(epoch, lr):
        #     if epoch < 2:
        #         return 0.01
        #     elif epoch < 4:
        #         return 0.001
        #     else:
        #         return lr * np.exp(-0.1)

        return ds_train, ds_val, ds_test, class_weights
    
    ###########################################################################
    # SpokeN-100
    ###########################################################################
    elif dataset == "spoken":
        dataloader_spoken = SpokenDataLoader("datasets/spokeN-100/")
        data = dataloader_spoken.load_waveform(label_type="number")
        ds_train, ds_val, ds_test = data["train"], data["val"], data["test"]

        # def lr_schedule(epoch, lr):
        #     if epoch < 25:
        #         return 0.0001
        #     elif epoch < 50:
        #         return 0.00005
        #     else:
        #         return 0.00001

        return ds_train, ds_val, ds_test, None

    elif dataset == "airway":
        dataloader_airway = AIrwayDataLoader("../../datasets/airway_database/", params)
        ds_train, ds_val, ds_test = dataloader_airway.load_data()

        # class weights are hard-coded to avoid re-calculating them every time
        # they aredetermined by the following out-commented code
        # class_weights = get_class_weights(ds_train, params["num_classes"])
        class_weights = None

        # class_weights = {
        #     0: 18.04306758674693,
        #     1: 30.182071223905034,
        #     2: 13.962808665353734,
        #     3: 27.078516342269555,
        #     4: 22.421334306391778,
        #     5: 0.6243209374788322,
        #     6: 0.19392789812991298
        # }

        return ds_train, ds_val, ds_test, class_weights


    elif dataset == "motion_sense":
        raise NotImplementedError("MotionSense dataset is not implemented yet.")
    
    elif dataset == "daliac":
        dataloader_daliac = DaliacDataLoader(path, return_one_hot=return_one_hot)
        ds_train, ds_val, ds_test = dataloader_daliac.load_dataset()
        return ds_train, ds_val, ds_test, None
    
    elif dataset == "toy_admos":
        dataloader_toy_admos = ToyAdmosDataloader("datasets/ToyADMOS/")
        ds_train, ds_val, ds_test = dataloader_toy_admos.load_dataset()
        return ds_train, ds_val, ds_test

    elif dataset == "corscience":
        # TODO: remove batch_size
        dataloader_corscience = CorscienceDataLoader("datasets/EKG Daten/DataSegmented/", batch_size=32)
        ds_train, ds_val, ds_test = dataloader_corscience.load_dataset()
        return ds_train, ds_val, ds_test
    
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


# def _resample_func(x, samples):
#     x = signal.resample(x, samples, axis=0)
#     return x[..., np.newaxis]


# def _normalize(data, label):
#     data /= tf.math.reduce_max(tf.abs(data), axis=0)
#     return data, label
