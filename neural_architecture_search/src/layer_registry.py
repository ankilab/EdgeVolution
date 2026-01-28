"""
Layer Registry for Neural Architecture Search.

This module provides a central registry for all layers that can be used in the search space.
It ensures that only implemented layers can be referenced in search space configurations,
providing early validation and helpful error messages.

Usage:
    # Register a custom layer or factory function
    @LayerRegistry.register()
    def get_my_custom_block(filters, activation):
        ...

    # Register with custom name and metadata
    @LayerRegistry.register(name='MyBlock', metadata={'category': 'feature_extraction'})
    def get_my_block(...):
        ...

    # Check if a layer exists
    LayerRegistry.exists('Dense')  # True

    # Get a layer
    layer_fn = LayerRegistry.get('Dense')

    # List all available layers
    LayerRegistry.list_available()
"""

from typing import Callable, Dict, Any, List, Optional, Type, Union
from difflib import get_close_matches
import importlib
import pkgutil


class LayerNotFoundError(Exception):
    """Raised when a layer is not found in the registry."""

    def __init__(self, missing: str, available: List[str]):
        self.missing = missing
        self.available = available
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        suggestions = get_close_matches(self.missing, self.available, n=3, cutoff=0.4)
        suggestion_text = ""
        if suggestions:
            suggestion_text = f"\n\nDid you mean one of these?\n" + "\n".join(
                f"  - {s}" for s in suggestions
            )

        return (
            f"Layer '{self.missing}' is not registered.{suggestion_text}\n\n"
            f"Available layers ({len(self.available)}):\n"
            + "\n".join(f"  - {name}" for name in sorted(self.available)[:30])
            + ("\n  ..." if len(self.available) > 30 else "")
            + "\n\nTo register a custom layer, use @LayerRegistry.register()"
        )


class LayerRegistry:
    """
    Central registry for all layers usable in the neural architecture search space.

    The registry tracks:
    - What layers are implemented and available
    - Metadata about each layer (source, category, etc.)
    - Provides validation and lookup functionality

    Layers can be:
    - Keras built-in layers (auto-registered)
    - Custom layer classes
    - Factory functions that return layers
    """

    _layers: Dict[str, Callable] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazily initialize the registry with built-in layers."""
        if not cls._initialized:
            cls._register_builtins()
            cls._initialized = True

    @classmethod
    def _register_builtins(cls) -> None:
        """Auto-register common Keras and third-party layers."""
        # TensorFlow Keras layers
        try:
            import tensorflow as tf

            keras_layers = [
                # Core layers
                "Dense",
                "Activation",
                "Dropout",
                "Flatten",
                "Reshape",
                "Permute",
                # Convolutional layers
                "Conv1D",
                "Conv2D",
                "Conv3D",
                "DepthwiseConv1D",
                "DepthwiseConv2D",
                "SeparableConv1D",
                "SeparableConv2D",
                # Pooling layers
                "MaxPooling1D",
                "MaxPooling2D",
                "AveragePooling1D",
                "AveragePooling2D",
                "GlobalAveragePooling1D",
                "GlobalAveragePooling2D",
                "GlobalMaxPooling1D",
                "GlobalMaxPooling2D",
                # Normalization layers
                "BatchNormalization",
                "LayerNormalization",
                # Regularization
                "Dropout",
                "SpatialDropout1D",
                "SpatialDropout2D",
                # Other
                "Resizing",
            ]
            for name in keras_layers:
                layer_class = getattr(tf.keras.layers, name, None)
                if layer_class:
                    cls._layers[name] = layer_class
                    cls._metadata[name] = {"source": "keras", "trainable": True}
        except ImportError:
            pass

        # Kapre layers (audio processing)
        try:
            from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank

            kapre_layers = {
                "STFT": STFT,
                "Magnitude": Magnitude,
                "MagnitudeToDecibel": MagnitudeToDecibel,
                "ApplyFilterbank": ApplyFilterbank,
            }
            for name, layer_class in kapre_layers.items():
                cls._layers[name] = layer_class
                cls._metadata[name] = {"source": "kapre", "category": "preprocessing"}
        except ImportError:
            pass

        # TensorFlow Addons
        try:
            from tensorflow_addons.layers import InstanceNormalization

            cls._layers["InstanceNormalization"] = InstanceNormalization
            cls._metadata["InstanceNormalization"] = {
                "source": "tensorflow_addons",
                "category": "normalization",
            }
        except ImportError:
            pass

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to register a layer or factory function.

        Args:
            name: Optional custom name for the layer. If not provided,
                  uses the function/class name.
            metadata: Optional metadata dict (category, source, etc.)

        Returns:
            Decorator function

        Example:
            @LayerRegistry.register()
            def get_conv2d_block(filters, kernel_size):
                ...

            @LayerRegistry.register(name='SE_Block', metadata={'category': 'attention'})
            class SqueezeExcitation(tf.keras.layers.Layer):
                ...
        """
        cls._ensure_initialized()

        def decorator(func_or_class: Union[Callable, Type]) -> Union[Callable, Type]:
            key = name if name is not None else func_or_class.__name__
            cls._layers[key] = func_or_class
            cls._metadata[key] = metadata or {}
            return func_or_class

        return decorator

    @classmethod
    def register_direct(
        cls,
        name: str,
        layer: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a layer directly without using the decorator.

        Args:
            name: Name to register the layer under
            layer: The layer class or factory function
            metadata: Optional metadata dict
        """
        cls._ensure_initialized()
        cls._layers[name] = layer
        cls._metadata[name] = metadata or {}

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get a registered layer by name.

        Args:
            name: The layer name

        Returns:
            The layer class or factory function

        Raises:
            LayerNotFoundError: If the layer is not registered
        """
        cls._ensure_initialized()

        # Handle special case of "()" suffix (e.g., "Magnitude()")
        clean_name = name.rstrip("()")

        if clean_name not in cls._layers:
            raise LayerNotFoundError(clean_name, list(cls._layers.keys()))

        return cls._layers[clean_name]

    @classmethod
    def exists(cls, name: str) -> bool:
        """
        Check if a layer is registered.

        Args:
            name: The layer name

        Returns:
            True if the layer exists, False otherwise
        """
        cls._ensure_initialized()
        clean_name = name.rstrip("()")
        return clean_name in cls._layers

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered layer.

        Args:
            name: The layer name

        Returns:
            Metadata dict for the layer
        """
        cls._ensure_initialized()
        clean_name = name.rstrip("()")
        return cls._metadata.get(clean_name, {})

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered layer names.

        Returns:
            Sorted list of layer names
        """
        cls._ensure_initialized()
        return sorted(cls._layers.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """
        List all layers in a given category.

        Args:
            category: The category to filter by

        Returns:
            List of layer names in that category
        """
        cls._ensure_initialized()
        return sorted(
            name
            for name, meta in cls._metadata.items()
            if meta.get("category") == category
        )

    @classmethod
    def list_by_source(cls, source: str) -> List[str]:
        """
        List all layers from a given source.

        Args:
            source: The source to filter by (e.g., 'keras', 'custom')

        Returns:
            List of layer names from that source
        """
        cls._ensure_initialized()
        return sorted(
            name
            for name, meta in cls._metadata.items()
            if meta.get("source") == source
        )

    @classmethod
    def discover_layers(
        cls, package: str = "neural_architecture_search.src.search_space_modules"
    ) -> List[str]:
        """
        Auto-discover and import all modules in a package to trigger @register decorators.

        This is useful for automatically registering all custom layers defined
        in the search_space_modules directory.

        Args:
            package: The package path to scan for modules

        Returns:
            List of discovered module names
        """
        cls._ensure_initialized()
        discovered = []

        try:
            package_module = importlib.import_module(package)
            for _, module_name, _ in pkgutil.iter_modules(package_module.__path__):
                full_name = f"{package}.{module_name}"
                try:
                    importlib.import_module(full_name)
                    discovered.append(module_name)
                except ImportError as e:
                    print(f"Warning: Could not import {full_name}: {e}")
        except ImportError as e:
            print(f"Warning: Could not import package {package}: {e}")

        return discovered

    @classmethod
    def clear(cls) -> None:
        """Clear all registered layers. Mainly useful for testing."""
        cls._layers.clear()
        cls._metadata.clear()
        cls._initialized = False

    @classmethod
    def summary(cls) -> str:
        """
        Get a summary of registered layers.

        Returns:
            Formatted string summary
        """
        cls._ensure_initialized()

        # Group by source
        by_source: Dict[str, List[str]] = {}
        for name, meta in cls._metadata.items():
            source = meta.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(name)

        lines = [f"LayerRegistry: {len(cls._layers)} layers registered\n"]
        for source in sorted(by_source.keys()):
            layers = sorted(by_source[source])
            lines.append(f"  {source} ({len(layers)}):")
            lines.append(f"    {', '.join(layers[:10])}")
            if len(layers) > 10:
                lines.append(f"    ... and {len(layers) - 10} more")

        return "\n".join(lines)
