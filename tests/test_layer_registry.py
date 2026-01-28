"""Tests for the LayerRegistry class."""

import unittest

from neural_architecture_search.src.layer_registry import (
    LayerRegistry,
    LayerNotFoundError,
)


class TestLayerRegistry(unittest.TestCase):
    """Test cases for LayerRegistry."""

    def setUp(self):
        """Reset the registry before each test."""
        LayerRegistry.clear()

    def test_register_decorator(self):
        """Test registering a layer with the decorator."""

        @LayerRegistry.register()
        def my_custom_layer(filters):
            return f"layer with {filters} filters"

        self.assertTrue(LayerRegistry.exists("my_custom_layer"))
        self.assertEqual(
            LayerRegistry.get("my_custom_layer")(32), "layer with 32 filters"
        )

    def test_register_with_custom_name(self):
        """Test registering with a custom name."""

        @LayerRegistry.register(name="CustomBlock")
        def get_block():
            return "block"

        self.assertTrue(LayerRegistry.exists("CustomBlock"))
        self.assertFalse(LayerRegistry.exists("get_block"))

    def test_register_with_metadata(self):
        """Test registering with metadata."""

        @LayerRegistry.register(metadata={"category": "conv", "source": "custom"})
        def conv_layer():
            pass

        metadata = LayerRegistry.get_metadata("conv_layer")
        self.assertEqual(metadata["category"], "conv")
        self.assertEqual(metadata["source"], "custom")

    def test_register_direct(self):
        """Test direct registration without decorator."""

        def my_layer():
            return "direct"

        LayerRegistry.register_direct("DirectLayer", my_layer, {"source": "test"})

        self.assertTrue(LayerRegistry.exists("DirectLayer"))
        self.assertEqual(LayerRegistry.get("DirectLayer")(), "direct")

    def test_get_nonexistent_raises_error(self):
        """Test that getting a non-existent layer raises LayerNotFoundError."""
        with self.assertRaises(LayerNotFoundError) as ctx:
            LayerRegistry.get("NonExistentLayer")

        self.assertEqual(ctx.exception.missing, "NonExistentLayer")
        self.assertIn("NonExistentLayer", str(ctx.exception))

    def test_handles_parentheses_suffix(self):
        """Test that layer names with () suffix are handled correctly."""

        @LayerRegistry.register()
        def TestLayer():
            return "test"

        # Should work with or without parentheses
        self.assertTrue(LayerRegistry.exists("TestLayer"))
        self.assertTrue(LayerRegistry.exists("TestLayer()"))
        self.assertIsNotNone(LayerRegistry.get("TestLayer()"))

    def test_list_available(self):
        """Test listing available layers."""

        @LayerRegistry.register()
        def layer_a():
            pass

        @LayerRegistry.register()
        def layer_b():
            pass

        available = LayerRegistry.list_available()
        self.assertIn("layer_a", available)
        self.assertIn("layer_b", available)

    def test_list_by_category(self):
        """Test filtering layers by category."""

        @LayerRegistry.register(metadata={"category": "conv"})
        def conv1():
            pass

        @LayerRegistry.register(metadata={"category": "conv"})
        def conv2():
            pass

        @LayerRegistry.register(metadata={"category": "dense"})
        def dense1():
            pass

        conv_layers = LayerRegistry.list_by_category("conv")
        self.assertEqual(len(conv_layers), 2)
        self.assertIn("conv1", conv_layers)
        self.assertIn("conv2", conv_layers)
        self.assertNotIn("dense1", conv_layers)

    def test_builtin_keras_layers_registered(self):
        """Test that built-in Keras layers are auto-registered."""
        # Force initialization
        LayerRegistry._ensure_initialized()

        # Check some common Keras layers
        self.assertTrue(LayerRegistry.exists("Dense"))
        self.assertTrue(LayerRegistry.exists("Conv2D"))
        self.assertTrue(LayerRegistry.exists("BatchNormalization"))
        self.assertTrue(LayerRegistry.exists("GlobalAveragePooling2D"))

    def test_summary(self):
        """Test the summary method."""

        @LayerRegistry.register(metadata={"source": "test"})
        def test_layer():
            pass

        summary = LayerRegistry.summary()
        self.assertIn("LayerRegistry", summary)
        self.assertIn("layers registered", summary)


class TestLayerNotFoundError(unittest.TestCase):
    """Test cases for LayerNotFoundError."""

    def test_error_message_contains_missing_name(self):
        """Test that error message contains the missing layer name."""
        error = LayerNotFoundError("MissingLayer", ["Dense", "Conv2D"])
        self.assertIn("MissingLayer", str(error))

    def test_error_suggests_similar_names(self):
        """Test that error suggests similar layer names."""
        error = LayerNotFoundError("Dene", ["Dense", "Conv2D", "Dropout"])
        self.assertIn("Dense", str(error))  # Should suggest Dense

    def test_error_lists_available_layers(self):
        """Test that error lists available layers."""
        error = LayerNotFoundError("Missing", ["Dense", "Conv2D"])
        self.assertIn("Dense", str(error))
        self.assertIn("Conv2D", str(error))


if __name__ == "__main__":
    unittest.main()
