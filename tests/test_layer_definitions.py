"""Tests for neural_architecture_search.src.layer_definitions module."""

import unittest

from neural_architecture_search.src.layer_registry import LayerRegistry

try:
    from neural_architecture_search.src.layer_definitions import (
        instantiate_layer,
        validate_chromosome,
        get_classification_layer,
    )
    LAYER_DEFS_AVAILABLE = True
except ImportError:
    LAYER_DEFS_AVAILABLE = False

try:
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper mock functions used across tests
# ---------------------------------------------------------------------------

def _mock_layer(**kwargs):
    """Mock layer factory that returns its keyword arguments."""
    return {"mock": True, **kwargs}


def _mock_no_arg_layer():
    """Mock layer factory that takes no arguments."""
    return {"mock_no_arg": True}


# ---------------------------------------------------------------------------
# TestInstantiateLayer
# ---------------------------------------------------------------------------

@unittest.skipUnless(LAYER_DEFS_AVAILABLE, "layer_definitions requires tensorflow")
class TestInstantiateLayer(unittest.TestCase):
    """Tests for the instantiate_layer function."""

    def setUp(self):
        LayerRegistry.clear()

    # 1. Basic registry look-up
    def test_instantiation_via_registry(self):
        """A function registered via @LayerRegistry.register() is found and called."""

        @LayerRegistry.register(name="MockLayer")
        def mock_layer_fn(**kwargs):
            return "registry_hit"

        gene = {"f_name": "MockLayer"}
        result = instantiate_layer(gene, "test_layer")
        self.assertEqual(result, "registry_hit")

    # 2. f_name ending with "()" -> no-arg call
    def test_instantiation_no_arg_suffix(self):
        """When f_name ends with '()', the function is called with no arguments."""

        @LayerRegistry.register(name="NoArgFunc")
        def no_arg_func():
            return "no_args_called"

        gene = {"f_name": "NoArgFunc()"}
        result = instantiate_layer(gene, "test_layer")
        self.assertEqual(result, "no_args_called")

    # 3. Extra params (non-meta keys) are forwarded
    def test_instantiation_with_params(self):
        """Only non-meta params (not 'layer' or 'f_name') are forwarded."""

        @LayerRegistry.register(name="ParamFunc")
        def param_func(**kwargs):
            return kwargs

        gene = {
            "layer": "some_layer",
            "f_name": "ParamFunc",
            "filters": 32,
            "kernel_size": 3,
        }
        result = instantiate_layer(gene, "test_layer")
        self.assertEqual(result, {"filters": 32, "kernel_size": 3})
        self.assertNotIn("layer", result)
        self.assertNotIn("f_name", result)

    # 4. Missing f_name -> ValueError
    def test_instantiation_missing_f_name(self):
        """A gene without 'f_name' raises ValueError."""
        gene = {"layer": "oops"}
        with self.assertRaises(ValueError):
            instantiate_layer(gene, "test_layer")


# ---------------------------------------------------------------------------
# TestValidateChromosome
# ---------------------------------------------------------------------------

@unittest.skipUnless(LAYER_DEFS_AVAILABLE, "layer_definitions requires tensorflow")
class TestValidateChromosome(unittest.TestCase):
    """Tests for the validate_chromosome function."""

    def setUp(self):
        LayerRegistry.clear()

    def test_valid_chromosome_passes(self):
        """A chromosome whose f_names are all registered returns True."""

        @LayerRegistry.register(name="ValidLayer")
        def valid_layer():
            pass

        chromosome = [
            {"layer": "block_1", "f_name": "ValidLayer"},
        ]
        self.assertTrue(validate_chromosome(chromosome))

    def test_invalid_chromosome_raises_value_error(self):
        """A chromosome referencing an unregistered f_name raises ValueError."""
        chromosome = [
            {"layer": "block_1", "f_name": "TotallyUnknownLayer"},
        ]
        with self.assertRaises(ValueError):
            validate_chromosome(chromosome)


# ---------------------------------------------------------------------------
# TestGetClassificationLayer
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    LAYER_DEFS_AVAILABLE and TF_AVAILABLE,
    "Requires tensorflow and layer_definitions",
)
class TestGetClassificationLayer(unittest.TestCase):
    """Tests for the get_classification_layer function."""

    def test_multiclass(self):
        """num_classes=10 with softmax returns Dense(10, activation='softmax')."""
        layer = get_classification_layer(num_classes=10, top_activation="softmax")
        self.assertIsInstance(layer, Dense)
        config = layer.get_config()
        self.assertEqual(config["units"], 10)
        self.assertEqual(config["activation"], "softmax")

    def test_binary_classification(self):
        """num_classes=2 with sigmoid returns Dense(1, activation='sigmoid')."""
        layer = get_classification_layer(num_classes=2, top_activation="sigmoid")
        self.assertIsInstance(layer, Dense)
        config = layer.get_config()
        self.assertEqual(config["units"], 1)
        self.assertEqual(config["activation"], "sigmoid")


if __name__ == "__main__":
    unittest.main()
