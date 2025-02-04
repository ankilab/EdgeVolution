class BaseDataLoader:
    def load_dataset(self):
        """Method to load and return the dataset.
        Should return: (ds_train, ds_val, ds_test, class_weights)
        """
        raise NotImplementedError("You must implement the `load_dataset` method.")