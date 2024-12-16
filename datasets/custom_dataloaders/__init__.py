import os
import importlib.util

def dynamic_import(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".py")  and filename != "__init__.py":
            module_name = filename[:-3]  # Remove ".py" extension
            module_path = os.path.join(directory, filename)

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

# Automatically load all dataloaders in the custom_dataloaders/ directory
dynamic_import("datasets/custom_dataloaders")