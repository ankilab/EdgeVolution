from datasets.src import *
from datasets.custom_dataloaders import *
from datasets.utils.registry import DATASET_REGISTRY, load_user_dataloaders

def load_dataset(dataset_name, **kwargs):
    """
    Load the specified dataset.

    Args:
        dataset_name (str): The name of the dataset.
        **kwargs: Additional arguments for the dataloader.
        
    Returns:
        ds_train, ds_val, ds_test, class_weights: The train, validation, and test datasets and class weights.
    
    Raises:
        ValueError: If the given dataset is not available.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset {dataset_name} is not registered. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    dataloader_cls = DATASET_REGISTRY[dataset_name]
    dataloader = dataloader_cls(**kwargs)  # Pass additional arguments to the DataLoader
    return dataloader.load_dataset()

# Dynamically load custom dataloaders
load_user_dataloaders("custom_dataloaders")