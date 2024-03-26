import tensorflow as tf
import librosa
import numpy as np
import sys
import os
from pathlib import Path

# Dataloader adapted from 
# https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/common.py

class ToyAdmosDataloader:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load_data(paths):
        data = []
        for path in paths:
            frames = 5
            n_mels = 128
            dims = n_mels * frames
            power = 2.0


            y, sr = librosa.load(path, sr=None, mono=False)
            
            mel_spec = librosa.feature.melspectrogram(y=y, 
                                                      sr=sr, 
                                                      n_fft=1024,
                                                      hop_length=512, 
                                                      n_mels=n_mels, 
                                                      power=power)

            # convert melspectrogram to log mel energy
            log_mel_spectrogram = 20.0 / power * np.log10(mel_spec + sys.float_info.epsilon)

            # take central part only
            log_mel_spectrogram = log_mel_spectrogram[:,50:250]

            # calculate total vector size
            vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

            # skip too short clips
            if vector_array_size < 1:
                return np.empty((0, dims))

            # generate feature vectors by concatenating multiframes
            vector_array = np.zeros((vector_array_size, dims))
            for t in range(frames):
                vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

            data.append(vector_array)
    
        return np.array(data)

    def load_dataset(self):
        # load train data
        train_files = [p for p in os.listdir(self.path / "train") if p.startswith("train")]
        train_paths = [str(self.path / "train" / file) for file in train_files]
        train_data = self.load_data(train_paths)

        # split train into train and val
        split = int(0.8 * len(train_data))
        val_data = tf.data.Dataset.from_tensor_slices(train_data[split:])
        train_data = tf.data.Dataset.from_tensor_slices(train_data[:split])

        # load test data
        test_files = [p for p in os.listdir(self.path / "test") if p.startswith("test")]
        test_paths = [str(self.path / "test" / file) for file in test_files]
        test_data = tf.data.Dataset.from_tensor_slices(self.load_data(test_paths))

        return train_data, val_data, test_data

