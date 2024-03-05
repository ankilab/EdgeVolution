import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob


class DaliacDataLoader:
    """
    DaLiAc dataset loader.
    """
    def __init__(self, data_path=None, window_size=2048, batch_size=32, overlap_percent=50):
        self.window_size = window_size
        self.batch_size = batch_size
        self.overlap_percent = overlap_percent
        self.num_channels = 1  # Assuming the magnitude of X, Y, Z axes

        if data_path is None:
            # find directory "datasets/" within the project directory
            data_path = glob.glob(f"../**/daliac/", recursive=True)[0]

        # Split the file paths into train and test
        self.train_file_paths = [data_path + f"dataset_{i}.txt" for i in range(1, 16)]
        self.val_file_paths = [data_path + f"dataset_{i}.txt" for i in range(16, 18)]
        self.test_file_paths = [data_path + f"dataset_{i}.txt" for i in range(18, 20)]

    @staticmethod
    def _load_accelerometer_data(file_path):
        """
        Load accelerometer data from the given file path.
        """
        df = pd.read_csv(str(file_path), sep=",", header=None)
        acc_data = df.iloc[:, [0, 1, 2]]

        # calculate magnitude
        magnitude = np.sqrt(np.square(acc_data).sum(axis=1))

        # normalize magnitude
        magnitude = (magnitude - magnitude.mean()) / magnitude.std()

        return magnitude, df.iloc[:, -1]

    @staticmethod
    def _most_frequent_label(labels):
        """
        Return the most frequent label in the given labels.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        most_frequent_index = np.argmax(counts)
        return unique_labels[most_frequent_index]

    def _load_data(self, file_paths):
        """
        Generator function to create the dataset.
        """
        X, y = [], []
        for file_path in file_paths:
            data, labels = self._load_accelerometer_data(file_path)

            # iterate over data and create windows --> add to X
            for i in range(0, len(data) - self.window_size, int(self.window_size * (1 - self.overlap_percent / 100))):
                X.append(data[i:i + self.window_size])
                y.append(self._most_frequent_label(labels[i:i + self.window_size]) - 1)

        return np.array(X)[..., None], np.array(y)

    def load_dataset(self):
        """
        Load the DaLiAc dataset.
        """
        ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.train_file_paths))).batch(self.batch_size)
        ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.val_file_paths))).batch(self.batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.test_file_paths))).batch(self.batch_size)

        return ds_train, ds_val, ds_test
