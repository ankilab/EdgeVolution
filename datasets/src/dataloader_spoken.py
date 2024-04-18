import tensorflow as tf
import librosa
from pathlib import Path
import numpy as np
from scipy import signal
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display


class SpokenDataLoader:
    @staticmethod
    def check_params(params: dict):
        """
        Check if the given parameters are valid to load this dataset.

        Args:
            params (dict): The parameters to check.

        Raises:
            ValueError: If a parameter is missing.
        """
        params_needed = {"num_classes": [4, 100], "sample_rate": 8000, "input_shape": [8000, 1]}
        
        for param in params_needed.keys():
            if param not in params.keys():
                raise ValueError(f"Parameter {param} is missing. Typical value: {params_needed[param]}. Please add it to the params dict.")

    def __init__(self, path: str):
        """
        Initialize the SpokenDataLoader.

        Args:
            data_split (pd.DataFrame): Dataframe containing the data split.
        """
        path = Path(path)
        self.paths = {"German": path / "german_numbers", 
                 "French": path / "french_numbers", 
                 "English": path / "english_numbers", 
                 "Mandarin": path / "mandarin_numbers"}
   
        df = self._get_df_metadata()
        speakers = df["speaker"].unique()

        # split the data into train, val and test --> all speaker for training, 1 speaker for validation and 1 speaker for testing --> add split column to df
        df["split"] = "train"
        df.loc[df["speaker"] == speakers[0], "split"] = "validation"
        df.loc[df["speaker"] == speakers[1], "split"] = "test"

        self.data_split = df
        self.label_type = None

        self.language_classes = {"English": 0, "Mandarin": 1, "German": 2, "French": 3}

    def _get_df_metadata(self):
        # create a df with all the file paths, language, speaker and number
        df = pd.DataFrame(columns=["file_path", "language", "speaker", "number"])
        for language in self.paths.keys():
            for speaker in self.paths[language].iterdir():
                if not speaker.is_dir():
                     continue
                for file in speaker.iterdir():
                        df = pd.concat([df, pd.DataFrame({"file_path": [file], "language": [language], "speaker": [speaker.stem], "number": [file.stem]})])
        return df

    def _get_label(self, df_row: pd.Series):
        """
        Get the label for a given dataframe row.

        Args:
            df_row (pd.Series): The dataframe row.

        Returns:
            The label for the given row.
        """
        if self.label_type == "language":
            label = self.language_classes.get(df_row["language"])
            return to_categorical(label, num_classes=len(self.language_classes))
        elif self.label_type == "number":
            return to_categorical(df_row["number"], num_classes=100)

    @staticmethod
    def _resample_func(x, samples):
        """
        Resample the given signal to the given number of samples.

        Args:
            x: The signal to be resampled.
            samples: The number of samples to resample to.

        Returns:
            The resampled signal.
        """
        x = signal.resample(x, samples, axis=0)
        return x

    def _load_data(self, df: pd.DataFrame, spectrogram=False, channel_dim=False):
        """
        Load the data from the dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            spectrogram (bool, optional): Whether to compute spectrogram. Defaults to False.
            channel_dim (bool, optional): Whether to include channel dimension in the spectrogram data. Defaults to False.

        Returns:
            The loaded data and labels.
        """
        X, y = [], []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
            file_path = row["file_path"]
            label = self._get_label(row)
            data, _ = librosa.load(file_path, sr=44100)

            # Resample to 8000 Hz
            data = self._resample_func(data, 8000)

            if spectrogram:
                # Create spectrogram
                data = librosa.feature.melspectrogram(y=data, sr=8000, n_fft=1024, hop_length=256, n_mels=128)
                data = librosa.power_to_db(data, ref=np.max)
                if channel_dim == 1:
                    data = data[..., np.newaxis]
                elif channel_dim == 3:
                    data = np.repeat(data[..., np.newaxis], 3, -1)
                elif channel_dim == "transformer":
                    # switch frequency and time axes
                    data = data.transpose()

            X.append(data)
            y.append(label)

        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

        return np.array(X), np.array(y)

    def load_waveform(self, label_type="language"):
            """
            Loads raw waveform data from the dataset.

            Args:
                label_type (str): The type of labels to load. Default is "language".

            Returns:
                dict: A dictionary containing the loaded datasets for training, validation, and testing.
                      The keys are "train", "val", and "test", respectively.
                      The values are TensorFlow datasets.

            """
            self.label_type = label_type

            ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "train"])))
            ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "validation"])))
            ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "test"])))

            return {"train": ds_train, "val": ds_val, "test": ds_test}
    
    def load_spectrogram(self, label_type: str = "language", channel_dim: int = 0):
        """
        Loads spectrogram data from the dataset.

        Note: I removed .batch(self.batch_size) here!

        Args:
            label_type (str, optional): Type of labels to use. Defaults to "language".
            channel_dim (int, optional): The index of the channel dimension in the spectrogram data. Set to 0 to not set any channel dimension, and set to 1 or 3 to set channel dimension to 1 or 3. Defaults to 0.

        Returns:
            dict: A dictionary containing the loaded spectrogram datasets for training, validation, and testing.
                  The keys are "train", "val", and "test", respectively.
        """
        self.label_type = label_type

        ds_train = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "train"], spectrogram=True, channel_dim=channel_dim)))
        try:
            ds_val = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "validation"], spectrogram=True, channel_dim=channel_dim)))
        except:
            ds_val = None
        ds_test = tf.data.Dataset.from_tensor_slices((self._load_data(self.data_split[self.data_split["split"] == "test"], spectrogram=True, channel_dim=channel_dim)))

        return {"train": ds_train, "val": ds_val, "test": ds_test}

    