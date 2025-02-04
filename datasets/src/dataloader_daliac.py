from datasets.utils.registry import register_dataset
from datasets.src.base_dataloader import BaseDataLoader

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

@register_dataset("daliac")
class DaliacDataLoader(BaseDataLoader):
    """
    DaLiAc dataset loader.
    """
    def __init__(self, return_one_hot=True, window_size=1024, overlap_percent=50):
        self.window_size = window_size
        self.overlap_percent = overlap_percent
        self.num_channels = 1  # Assuming the magnitude of X, Y, Z axes
        self.return_one_hot = return_one_hot

        # go dynamically back in directory until folder "EdgeVolution" is reached
        folder = os.getcwd()
        while os.path.basename(folder) != "EdgeVolution":
            folder = os.path.dirname(folder)

        # go to the datasets folder
        data_path = os.path.join(folder, "datasets/data/daliac/")

        # Split the file paths into train and test
        self.train_file_paths = [data_path + f"dataset_{i}.txt" for i in range(1, 15)]
        self.val_file_paths = [data_path + f"dataset_{i}.txt" for i in range(15, 18)]
        self.test_file_paths = [data_path + f"dataset_{i}.txt" for i in range(18, 20)]
        
        # Classes clustered accordingly (see https://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.0075196.g005)
        self.classes_clustered = {"HOUSE": [5, 6], "REST": [1, 2, 3], "WALK": [7, 8, 9, 10], "BICYCLE": [11, 12], "RJ": [13], "WD": [4]} 
        
        # Now I assign new class labels to the classes
        self.classes = {"HOUSE": 0, "REST": 1, "WALK": 2, "BICYCLE": 3, "RJ": 4, "WD": 5}
        
    def load_dataset(self):
        """
        Load the DaLiAc dataset.
        """
        ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.train_file_paths))).shuffle(1000)
        ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.val_file_paths)))
        ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.test_file_paths)))

        return ds_train, ds_val, ds_test, None
        
    @staticmethod
    def _load_accelerometer_data(file_path):
        """
        Load accelerometer data from the given file path.
        """
        df = pd.read_csv(str(file_path), sep=",", header=None)

        acc_data = df.iloc[:, [0, 1, 2]] # Corresponds to X, Y, Z accelerometer data recorded at the wrist

        # calculate magnitude
        magnitude = np.sqrt(np.square(acc_data).sum(axis=1))

        # normalize magnitude
        magnitude = (magnitude - magnitude.mean()) / magnitude.std()

        return magnitude, df.iloc[:, -1]
    
    def _get_single_label(self, labels):
        """
        Get the label but only if one single label occurs. Otherwise return None.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            # check if label is in the clustered classes
            for key, value in self.classes_clustered.items():
                if unique_labels[0] in value:
                    return self.classes[key]
        return None

    def _load_data(self, file_paths):
        """
        Generator function to create the dataset.
        """
        X, y = [], []
        for file_path in file_paths:
            data, labels = self._load_accelerometer_data(file_path)

            # iterate over data and create windows --> add to X
            for i in range(0, len(data) - self.window_size, int(self.window_size * (1 - self.overlap_percent / 100))):
                _X = data[i:i + self.window_size]
                _y = self._get_single_label(labels[i:i + self.window_size])

                if _y is not None:
                    X.append(_X)
                    if self.return_one_hot:
                        _y = to_categorical(_y, num_classes=len(self.classes))
                    y.append(_y)

        return np.array(X)[..., None], np.array(y)
    
    @staticmethod
    def _load_complete_accelerometer_data(file_path):
        """
        Load accelerometer data from the given file path.
        """
        df = pd.read_csv(str(file_path), sep=",", header=None)

        acc_data = df.iloc[:, :-1]
        return acc_data, df.iloc[:, -1]
 
    
    def _load_complete_data(self, file_paths):
        """
        Load the complete data from the given file paths.
        """
        X, y = [], []
        for file_path in file_paths:
            data, labels = self._load_complete_accelerometer_data(file_path)
            
            for i in range(0, len(data) - self.window_size, int(self.window_size * (1 - self.overlap_percent / 100))):
                _X = data[i:i + self.window_size]
                _y = self._get_single_label(labels[i:i + self.window_size])

                if _y is not None:
                    X.append(_X)
                    if self.return_one_hot:
                        _y = to_categorical(_y-1, num_classes=13)
                    y.append(_y)

        # normalize X
        X = (X - np.mean(X)) / np.std(X)

        # shuffle X and y
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = np.array(X)[indices]
        y = np.array(y)[indices]

        return np.array(X)[..., None], np.array(y)
    
    def load_complete_dataset(self):
        """
        Load the DaLiAc dataset.
        """
        ds_train = tf.data.Dataset.from_tensor_slices((self._load_complete_data(self.train_file_paths)))
        ds_val = tf.data.Dataset.from_tensor_slices((self._load_complete_data(self.val_file_paths)))
        ds_test = tf.data.Dataset.from_tensor_slices((self._load_complete_data(self.test_file_paths)))

        return ds_train, ds_val, ds_test 
