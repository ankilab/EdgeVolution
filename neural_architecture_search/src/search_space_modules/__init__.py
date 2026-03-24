"""
Search Space Modules for Neural Architecture Search.

This package contains custom layer implementations and factory functions
used in the NAS search space. All layers should be registered with the
LayerRegistry to be available in search space configurations.

Usage:
    # Import to trigger registration of all custom layers
    from neural_architecture_search.src import search_space_modules

    # Or use LayerRegistry.discover_layers() for automatic discovery
    from neural_architecture_search.src.layer_registry import LayerRegistry
    LayerRegistry.discover_layers()
"""

# Import all modules to trigger @LayerRegistry.register() decorators
from neural_architecture_search.src.search_space_modules.conv2d_block import (
    get_conv2d_block,
    get_depthwise_conv2d_block,
)
from neural_architecture_search.src.search_space_modules.filterbank_layer import (
    get_filterbank_layer,
)
from neural_architecture_search.src.search_space_modules.sinc_conv_layer import (
    SincConv1D,
)

__all__ = [
    "get_conv2d_block",
    "get_depthwise_conv2d_block",
    "get_filterbank_layer",
    "SincConv1D",
]
