import importlib.util
import os

DATASET_REGISTRY = {}

def register_dataset(name):
    def wrapper(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper

def load_user_dataloaders(path_to_dataloaders_dir):
    if not os.path.exists(path_to_dataloaders_dir):
        return
    for filename in os.listdir(path_to_dataloaders_dir):
        if filename.endswith(".py"):
            module_path = os.path.join(path_to_dataloaders_dir, filename)
            spec = importlib.util.spec_from_file_location("user_dataloader", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
