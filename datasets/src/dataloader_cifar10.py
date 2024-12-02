from datasets.utils.registry import register_dataset
from datasets.src.base_dataloader import BaseDataLoader

import tensorflow as tf
import tensorflow_datasets as tfds
import os

@register_dataset("cifar10")
class Cifar10DataLoader(BaseDataLoader):
    def __init__(self) -> None:
        self.nb_classes = 10  # number of classes in the complete dataset

    def _prepare_dataset(self, ds):
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, self.nb_classes)))
        return ds
    
    def load_dataset(self):
        """
        Load the CIFAR-10 dataset.

        :return: ds_train, ds_val, ds_test
        """
        # go dynamically back in directory until folder "EvoNAS" is reached
        folder = os.getcwd()
        while os.path.basename(folder) != "EvoNAS":
            folder = os.path.dirname(folder)
        
        # go to the datasets folder
        folder = os.path.join(folder, "datasets/data")

        ds_train, ds_val, ds_test = tfds.load('cifar10', data_dir=folder, split=['train[:90%]', 'train[90%:]', 'test'],
                                                          as_supervised=True, download=True, with_info=False)
        
        ds_train = self._prepare_dataset(ds_train)
        ds_val = self._prepare_dataset(ds_val)
        ds_test = self._prepare_dataset(ds_test)
        
        return ds_train, ds_val, ds_test, None