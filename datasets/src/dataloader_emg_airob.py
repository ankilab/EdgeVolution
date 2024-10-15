import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from tensorflow.keras.utils import to_categorical


class EmgAirobDataLoader:
    """
    DaLiAc dataset loader.
    """
    def __init__(self, csv_file="merged_EMG_new_raw_data.csv", window_size=20, step_size=1):
        self.nb_classes = 7
        self.train_participants = [1, 2, 3]
        self.val_participants = [4, 5]
        self.test_participants = [6]

        # go dynamically back in directory until folder "EvoNAS" is reached
        folder = os.getcwd()
        while os.path.basename(folder) != "EvoNAS":
            folder = os.path.dirname(folder)
        
        # go to the datasets folder
        folder = os.path.join(folder, "datasets")
        csv_file = os.path.join(folder, csv_file)
        
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

        unique_labels = self.df["label"].unique() # Labels are strings
        self.df["label"] = self.df["label"].apply(lambda x: list(unique_labels).index(x)) # convert the labels to integers

        self.window_size = window_size
        self.step_size = step_size

        # Pre-process the data
        emg_data = self.df.iloc[:, 4:-1].values
        self.df.iloc[:, 4:-1] = emg_data.astype(float) / 128.0
    
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
        # Iterate over participants
        for participant in tqdm(participants, desc=f"Loading {split} data"):
            df_participant = self.df[self.df["participant"] == participant]

            # Extract labels and features as NumPy arrays early
            labels_participant = df_participant["label"].values
            features_participant = df_participant.iloc[:, 4:-1].values  # Get features as a NumPy array

            # Loop over windows with step size
            for i in tqdm(range(0, len(df_participant) - self.window_size, self.step_size), leave=False):
                window_labels = labels_participant[i:i+self.window_size]

                # Skip windows with inconsistent labels
                if len(np.unique(window_labels)) != 1:
                    continue

                features.append(features_participant[i:i+self.window_size, :])  # Append the feature window
                labels.append(window_labels[0])  # Append the consistent label

        # labels to one-hot encoding
        labels = to_categorical(labels, num_classes=self.nb_classes)

        return np.array(features), np.array(labels)

    def load_dataset(self):
        """
        Load the dataset.
        """
        x_train, y_train = self.load_data("train")
        x_val, y_val = self.load_data("val")
        x_test, y_test = self.load_data("test")

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1000)
        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        return ds_train, ds_val, ds_test
