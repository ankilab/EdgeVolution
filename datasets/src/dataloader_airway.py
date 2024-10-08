import tensorflow as tf
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm.notebook import tqdm
import multiprocessing

from scipy import signal
from sklearn.model_selection import train_test_split

class AIrwayDataLoader:
    """
    All events and event keys:
    {
        'cough': 592,
        'wheeze': 1593,
        'sneeze': 721,
        'throat_clear': 574,
        'silence': 2669,
        'breath_before_sneeze': 231,
        'irregular_breathing': 51,
        'rapid_breathing': 29,
        'wet_cough': 100,
        'dry_cough': 116,
        'throat_clearing': 46,
        'deviated_voice': 2535,
        'dry_swallow': 362,
        'speech': 50,
        'breath_before_wet_cough': 7
    }
    """
    def __init__(self, path, params, shuffle=True):
        self.path = Path(path)
        self.shuffle = shuffle

        self.window_size = params["window_size"]
        
        self.classes = {"cough": 0, 
                        "throat_clear": 1, 
                        "wheeze": 2, 
                        "sneeze": 3,
                        "breathing": 4,
                        "deviated_voice": 5,
                        "none": 6}
        
    def parse_tf_records(self, example_proto):
        feature_description = {
            'data': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    def decode(self, example_proto):
        data = tf.io.decode_raw(example_proto['data'], out_type=tf.float32)
        label = example_proto['label']
        return data, label

    def load_tf_records(self, filename):
        raw_dataset = tf.data.TFRecordDataset(filename)
        parsed_dataset = raw_dataset.map(self.parse_tf_records)
        parsed_dataset = parsed_dataset.map(self.decode)
        return parsed_dataset
    
    def one_hot(label):
        return tf.one_hot(label, 7)
    
    def _predicate(self, x, label):
        allowed_labels = tf.constant([0, 6])
        isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
        reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
        return tf.greater(reduced, tf.constant(0.))
    
    def balance_dataset(self, ds):
        # get class with the least samples and reduce all other classes to the same amount
        class_counts = {}
        for _, label in ds.as_numpy_iterator():
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        min_class = min(class_counts, key=class_counts.get)
        min_class_count = class_counts[min_class]

        balanced_ds = ds.filter(lambda x, label: tf.equal(label, min_class))
        for i in range(7):
            if i != min_class:
                ds_class = ds.filter
                balanced_ds = balanced_ds.concatenate(ds_class)

        return balanced_ds

    
    def load_data(self): 
        print("Loading data...")
        ds_train = self.load_tf_records(self.path / f"tf_records/airway_{self.window_size}_train.tfrecord")
        ds_val = self.load_tf_records(self.path / f"tf_records/airway_{self.window_size}_val.tfrecord")
        ds_test = self.load_tf_records(self.path / f"tf_records/airway_{self.window_size}_test.tfrecord")

        # balance the dataset
        # ds_train = self.balance_dataset(ds_train)

        # shuffle ds_train
        if self.shuffle:
            ds_train = ds_train.shuffle(1000)

        # ds_train = ds_train.filter(self._predicate)
        # ds_val = ds_val.filter(self._predicate)
        # ds_test = ds_test.filter(self._predicate)

        # one hot encoding
        n_classes = 7
        # ds_train = ds_train.map(lambda x, y: (x, tf.one_hot(y, n_classes)))
        # ds_val = ds_val.map(lambda x, y: (x, tf.one_hot(y, n_classes)))
        # ds_test = ds_test.map(lambda x, y: (x, tf.one_hot(y, n_classes)))

        return (ds_train, ds_val, ds_test)

    # def _downsample(self, data):
    #     num_original_samples = data.shape[0]
    #     num_target_samples = int((self.downsampling_rate / self.sr) * num_original_samples)
    #     return signal.resample(data, num_target_samples)
    
    # def _get_label(self, annotations, start, end, overlap_percent=0.3):
    #     events, _from, _to = annotations['event'], annotations['from'], annotations['to']
    #     for i in range(len(events)):
    #         event_start = _from[i]
    #         event_end = _to[i]
            
    #         # calculate the overlap between the event and the window
    #         overlap = max(0, min(end, event_end) - max(start, event_start))
    #         overlap_percentage = overlap / (end - start)
            
    #         # if the overlap is greater than overlap_percent, return the label
    #         if overlap_percentage >= overlap_percent:
    #             event = str(events[i].decode("utf-8"))

    #             if event == "dry_cough" or event == "wet_cough":
    #                 return self.classes["cough"]
    #             if "breath" in event:
    #                 return self.classes["breathing"]
    #             elif event in self.classes.keys():
    #                 return self.classes[event]
    #             else:
    #                 return self.classes["none"]
            
    #     return self.classes["none"]

    # def _load_hdf_file(self, path):
    #     print("Loading", path)

    #     # save the data and labels in the corresponding lists
    #     X_train, X_val, X_test = [], [], []
    #     y_train, y_val, y_test = [], [], []

    #     with h5py.File(path, 'r') as f:
    #         for key in tqdm(f.keys()):
    #             print(key)
    #             data, labels = [], []
    #             wav_data = f[key]['nsa'] # raw wav data
    #             wav_data = self._downsample(wav_data)  # downsample the data to the target sample rate

    #             # pad the data to be evenly divisible by the window size
    #             length = len(wav_data)
    #             if length % self.window_size != 0:
    #                 wav_data = np.pad(wav_data, (0, self.window_size - length % self.window_size), mode='constant', constant_values=0)

    #             # iterate over data with window size and hop
    #             for i in range(0, len(wav_data) - self.window_size, self.window_hop):
    #                 start = i
    #                 end = i + self.window_size
    #                 label = self._get_label(f[key], start, end)

    #                 data.append(wav_data[start:end])
    #                 labels.append(label)
        
    #     # split the data into training, validation and test set
    #     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    #     return X_train, X_val, X_test, y_train, y_val, y_test


    # def load_data(self):    
    #     X_train, X_val, X_test = [], [], []
    #     y_train, y_val, y_test = [], [], []

    #     # iterate over all hdf5 files in the directory
    #     for file in self.path.glob('*.hdf5'):

    #         # load hdf5 file 8each hdf5 file contains on single dataset which together form the whole airway database)
    #         X_train_, X_val_, X_test_, y_train_, y_val_, y_test_ = self._load_hdf_file(file)

    #         X_train.extend(X_train_)
    #         X_val.extend(X_val_)
    #         X_test.extend(X_test_)

    #         y_train.extend(y_train_)
    #         y_val.extend(y_val_)
    #         y_test.extend(y_test_)

    #         break

    #     # convert y labels to one hot encoding
    #     y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.classes))
    #     y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(self.classes))
    #     y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(self.classes))

    #     # create TensorFlow datasets        
    #     ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
    #     ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    #     ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    #     return (ds_train, ds_val, ds_test)
