from datasets.utils.registry import register_dataset
from datasets.src.base_dataloader import BaseDataLoader

@register_dataset("my_custom_dataset")
class MyCustomDataLoader(BaseDataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
        
        raise NotImplementedError("You must implement a custom dataloader.")

    def load_dataset(self):
        # Implementation for loading the custom dataset
        ds_train, ds_val, ds_test = None, None, None  # TensorFlow datasets
        return ds_train, ds_val, ds_test, None