import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import itertools


class CorscienceDataLoader:
    def __init__(self, data_dir: str, batch_size: int):
        # Define classes
        self.classes = {'nonshockable': 0, 'shockable': 1}
        self.shockable_classes = ['coarseVF', 'rapidVT', 'fineVF', 
                                  'VTabove180BPM', 'VTbelow180BPM']

        # Get all possible classes
        self.path = Path(data_dir)
        patient_ids = [x for x in self.path.iterdir() if x.is_dir()]

        self.all_classes = set()
        for patient in patient_ids:
            folders = os.listdir(patient)
            self.all_classes.update(folders)

    def _load_data(self, _class):
        data = []
        labels = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if _class in file:
                    data_path = os.path.join(root, file)
                    df = pd.read_csv(data_path, sep=',')
                    for column in df.columns:
                        column_data = df[column].values
                        for i in range(0, len(column_data), 2500):
                            data.append(column_data[i:i+2500])
                            if _class in self.shockable_classes:
                                labels.append(self.classes['shockable'])
                            else:
                                labels.append(self.classes['nonshockable'])
        return data, labels
    
    def load_dataset(self):
        data, labels = [], []
        for _class in self.all_classes:
            _data, _labels = self._load_data(_class)
            data.extend(_data)
            labels.extend(_labels)

        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42,
                                                            stratify=labels)

        # create tensorflow dataset
        ds_train = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(X_train, y_train),
                                                  output_types=(tf.float64, tf.int32),
                                                  output_shapes=((2500, ), ()))
        ds_val = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(X_val, y_val),
                                                output_types=(tf.float64, tf.int32),
                                                output_shapes=((2500, ), ()))

        return ds_train, ds_val, None