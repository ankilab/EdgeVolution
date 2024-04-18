import tensorflow as tf
import numpy as np
import itertools




class MotionSenseDataLoader:
    def __init__(self):
        pass

    def load_data(self):
        path = Path("datasets/motion_sense_accelerometer")
        classes = {'dws': 0, 'ups': 1, 'sit': 2, 'std': 3, 'wlk': 4, 'jog': 5}
        train_idxs, val_idxs, test_idxs = np.arange(0, 19), np.arange(19, 22), np.arange(22, 25)

        def _get_ds(data, labels, samples=4_000):
            ds = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(data, labels),
                                                output_types=(tf.float64, tf.int32),
                                                output_shapes=((samples, 1), ()))
            if classes_filter is not None:
                ds = ds.map(
                    lambda x, y: (x, tf.one_hot(tf.where(tf.equal(y, classes_filter))[0], len(classes_filter))[0]))
            else:
                ds = ds.map(lambda x, y: (x, tf.one_hot(y, len(classes))))
            return ds

        train_data, val_data, test_data = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for folder in os.listdir(path):
            class_path = path / folder
            for f in os.listdir(class_path):
                df = pd.read_csv(class_path / f)

                x, y, z = np.asarray(df['x']), np.asarray(df['y']), np.asarray(df['z'])
                magnitude = list(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
                magnitude = _resample_func(magnitude, samples)
                label = classes[folder[0:3]]

                idx = int(re.findall(r'\d+', f)[0])
                if idx in train_idxs:
                    train_data.append(magnitude)
                    train_labels.append(label)
                elif idx in val_idxs:
                    val_data.append(magnitude)
                    val_labels.append(label)
                else:
                    test_data.append(magnitude)
                    test_labels.append(label)

        ds_train = _get_ds(np.asarray(train_data), np.asarray(train_labels), samples)
        ds_val = _get_ds(np.asarray(val_data), np.asarray(val_labels), samples)
        ds_test = _get_ds(np.asarray(test_data), np.asarray(test_labels), samples)

        return ds_train, ds_val, ds_test