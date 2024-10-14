import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from tensorflow.keras.utils import to_categorical


class EmgAirobDataLoader:
    """
    DaLiAc dataset loader.
    """
    def __init__(self, csv_file="merged_EMG_new_raw_data.csv", window_size=20, step_size=1):
        self.nb_classes = 7
        self.train_participants = [1, 2, 3, 4]
        self.val_participants = [5]
        self.test_participants = [6]
        
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.df = self.df.rename(columns={'participant_y': 'participant', 'label2': 'label', 'control_y':'control', 'session_control_y':'session_control'})

        new_column_order = [
            'participant', 'control', 'session_control', 'timestamp',
            'EMG0', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7',
            'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12', 'EMG13', 'EMG14', 'EMG15',
            'label'
        ]

        # Reorder the columns
        self.df = self.df[new_column_order]

        # Omit all rows with NaN values
        self.df = self.df.dropna()

        unique_labels = self.df["label"].unique()
        self.df["label"] = self.df["label"].apply(lambda x: list(unique_labels).index(x))

        self.window_size = window_size
        self.step_size = step_size

        # Pre-process the data
        emg_data = self.df.iloc[:, 4:-1].values
        # # convert the data to float
        emg_data = emg_data.astype(float) / 128.0
    
    def load_data(self, split):
        """
        Load the data.
        """
        if split == "train":
            participants = self.train_participants
        elif split == "val":
            participants = self.val_participants
        elif split == "test":
            participants = self.test_participants
        else:
            raise ValueError(f"Invalid split: {split}")

        features = []
        labels = []
        for i in range(len(self.features)):
            if self.df["participant"].iloc[i] in participants:
                features.append(self.features[i])
                labels.append(self.labels[i])

        features = np.array(features)
        labels = to_categorical(labels, num_classes=self.nb_classes)

        return features, labels

    def load_dataset(self):
        """
        Load the dataset.
        """
        x_train, y_train = self.load_data("train")
        x_val, y_val = self.load_data("val")
        x_test, y_test = self.load_data("test")

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        return ds_train, ds_val, ds_test
