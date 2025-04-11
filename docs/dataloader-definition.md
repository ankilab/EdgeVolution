## Adding Datasets to the EdgeVolution Framework

To add more datasets to the EdgeVolution framework, follow these steps:

1. **Add Data Folder**: Place the folder containing your dataset into the `data/` directory.

2. **Create DataLoader**: 
    - Add a new file named `dataloader_<your_dataset_name>.py` to the `custom_dataloaders/` directory.
    - Inside this file, create a class that inherits from `BaseDataLoader` found in `src/base_dataloader.py`.
    - Implement your dataloader within this class. It should return `ds_train`, `ds_val`, and `ds_test` as TensorFlow datasets. Optionally, you can also return `class_weights` to be used during training.