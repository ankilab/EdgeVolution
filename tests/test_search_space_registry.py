"""Tests for the SearchSpaceRegistry class."""

import unittest
import tempfile
import os

import numpy as np
import yaml

from neural_architecture_search.src.search_space_registry import (
    SearchSpaceRegistry,
    SearchSpaceValidationError,
    ParameterSpec,
    LayerSpec,
)
from neural_architecture_search.src.layer_registry import LayerRegistry


class TestParameterSpec(unittest.TestCase):
    """Test cases for ParameterSpec."""

    def test_categorical_param(self):
        """Test categorical parameter specification."""
        spec = ParameterSpec(
            name="activation",
            param_type="categorical",
            values=["relu", "sigmoid", "tanh"],
        )
        self.assertEqual(spec.num_values, 3)
        self.assertEqual(spec.vector_size, 3)  # One-hot

    def test_discrete_param(self):
        """Test discrete parameter specification."""
        spec = ParameterSpec(
            name="filters",
            param_type="discrete",
            values=[8, 16, 32, 64],
        )
        self.assertEqual(spec.num_values, 4)
        self.assertEqual(spec.vector_size, 1)  # Normalized

    def test_continuous_param(self):
        """Test continuous parameter specification."""
        spec = ParameterSpec(
            name="dropout_rate",
            param_type="continuous",
            min_val=0.0,
            max_val=0.5,
        )
        self.assertEqual(spec.vector_size, 1)


class TestSearchSpaceRegistryBasic(unittest.TestCase):
    """Basic test cases for SearchSpaceRegistry."""

    def setUp(self):
        """Set up a minimal search space for testing."""
        self.layers = {
            "CONV": LayerSpec(
                name="CONV",
                f_name="Conv2D",
                category="feature",
                params={
                    "filters": ParameterSpec(
                        name="filters",
                        param_type="discrete",
                        values=[8, 16, 32, 64],
                    ),
                    "activation": ParameterSpec(
                        name="activation",
                        param_type="categorical",
                        values=["relu", "sigmoid"],
                    ),
                },
                successors=["CONV", "POOL"],
            ),
            "POOL": LayerSpec(
                name="POOL",
                f_name="GlobalAveragePooling2D",
                category="pooling",
                params={},
                successors=["DENSE"],
            ),
            "DENSE": LayerSpec(
                name="DENSE",
                f_name="Dense",
                category="classification",
                params={
                    "units": ParameterSpec(
                        name="units",
                        param_type="discrete",
                        values=[32, 64, 128],
                    ),
                },
                terminal=True,
            ),
        }

        self.categories = {
            "feature": {"successors": ["feature", "pooling"]},
            "pooling": {"successors": ["classification"]},
            "classification": {"successors": ["classification"], "terminal": True},
        }

        self.registry = SearchSpaceRegistry(
            layers=self.layers,
            categories=self.categories,
            start_layers=["CONV"],
            max_layers=8,
        )

    def test_initialization(self):
        """Test registry initialization."""
        self.assertEqual(len(self.registry.layers), 3)
        self.assertIn("CONV", self.registry.layer_names)
        self.assertIn("POOL", self.registry.layer_names)
        self.assertIn("DENSE", self.registry.layer_names)

    def test_get_successors(self):
        """Test getting valid successors for a layer."""
        conv_successors = self.registry.get_successors("CONV")
        self.assertIn("CONV", conv_successors)
        self.assertIn("POOL", conv_successors)

        pool_successors = self.registry.get_successors("POOL")
        self.assertIn("DENSE", pool_successors)

    def test_get_start_layers(self):
        """Test getting valid start layers."""
        start = self.registry.get_start_layers()
        self.assertEqual(start, ["CONV"])

    def test_is_terminal(self):
        """Test terminal layer detection."""
        self.assertTrue(self.registry.is_terminal("DENSE"))
        self.assertFalse(self.registry.is_terminal("CONV"))

    def test_vector_size(self):
        """Test that vector size is computed correctly."""
        # Should be (num_layers + 1) + sum of param sizes, times max_layers
        self.assertGreater(self.registry.vector_size, 0)


class TestEncodingDecoding(unittest.TestCase):
    """Test encoding and decoding functionality."""

    def setUp(self):
        """Set up a search space for encoding tests."""
        self.layers = {
            "CONV": LayerSpec(
                name="CONV",
                f_name="Conv2D",
                category="feature",
                params={
                    "filters": ParameterSpec(
                        name="filters",
                        param_type="discrete",
                        values=[8, 16, 32, 64],
                    ),
                },
                successors=["CONV", "POOL"],
            ),
            "POOL": LayerSpec(
                name="POOL",
                f_name="GlobalAveragePooling2D",
                category="pooling",
                params={},
                successors=["DENSE"],
            ),
            "DENSE": LayerSpec(
                name="DENSE",
                f_name="Dense",
                category="classification",
                params={
                    "units": ParameterSpec(
                        name="units",
                        param_type="discrete",
                        values=[32, 64, 128],
                    ),
                },
                terminal=True,
            ),
        }

        self.registry = SearchSpaceRegistry(
            layers=self.layers,
            categories={},
            start_layers=["CONV"],
            max_layers=8,
        )

    def test_encode_simple_chromosome(self):
        """Test encoding a simple chromosome."""
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 32},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
            {"layer": "DENSE", "f_name": "Dense", "units": 64},
        ]

        vector = self.registry.encode(chromosome)

        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape[0], self.registry.vector_size)
        self.assertEqual(vector.dtype, np.float32)

    def test_decode_returns_valid_chromosome(self):
        """Test that decoding produces a valid chromosome structure."""
        # Create a random vector
        vector = np.random.rand(self.registry.vector_size).astype(np.float32)

        chromosome = self.registry.decode(vector, enforce_rules=True)

        self.assertIsInstance(chromosome, list)
        for gene in chromosome:
            self.assertIn("layer", gene)
            self.assertIn("f_name", gene)

    def test_encode_decode_roundtrip(self):
        """Test that encoding then decoding preserves structure."""
        original = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 32},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
            {"layer": "DENSE", "f_name": "Dense", "units": 64},
        ]

        vector = self.registry.encode(original)
        decoded = self.registry.decode(vector, enforce_rules=False)

        # Check that layer types are preserved
        self.assertEqual(len(decoded), len(original))
        for orig, dec in zip(original, decoded):
            self.assertEqual(orig["layer"], dec["layer"])

    def test_decode_enforces_rules(self):
        """Test that decoding with rule enforcement produces valid sequences."""
        # Create multiple random vectors and check they all produce valid sequences
        for _ in range(10):
            vector = np.random.rand(self.registry.vector_size).astype(np.float32)
            chromosome = self.registry.decode(vector, enforce_rules=True)

            if len(chromosome) == 0:
                continue

            # First layer should be a valid start layer
            self.assertIn(chromosome[0]["layer"], self.registry.get_start_layers())

            # Each subsequent layer should be a valid successor
            for i in range(1, len(chromosome)):
                prev_layer = chromosome[i - 1]["layer"]
                curr_layer = chromosome[i]["layer"]
                valid_successors = self.registry.get_successors(prev_layer)
                self.assertIn(
                    curr_layer,
                    valid_successors,
                    f"{curr_layer} is not a valid successor of {prev_layer}",
                )


class TestYAMLLoading(unittest.TestCase):
    """Test loading search spaces from YAML."""

    def test_load_legacy_format(self):
        """Test loading the legacy gene_pool + rule_set format."""
        config = {
            "gene_pool": {
                "feature": [
                    {
                        "layer": "CONV",
                        "f_name": "Conv2D",
                        "filters": [8, 64, 8],
                    }
                ],
                "pooling": [
                    {
                        "layer": "GAP",
                        "f_name": "GlobalAveragePooling2D()",
                    }
                ],
            },
            "rule_set": {
                "Start": {"rule": ["CONV"]},
                "CONV": {"rule": ["CONV", "GAP"]},
                "GAP": {"rule": []},
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            registry = SearchSpaceRegistry.from_yaml(temp_path, validate=False)

            self.assertIn("CONV", registry.layer_names)
            self.assertIn("GAP", registry.layer_names)
            self.assertEqual(registry.get_start_layers(), ["CONV"])
        finally:
            os.unlink(temp_path)

    def test_load_unified_format(self):
        """Test loading the new unified format."""
        config = {
            "categories": {
                "feature": {"successors": ["feature", "pooling"]},
                "pooling": {"successors": [], "terminal": True},
            },
            "start": ["CONV"],
            "layers": {
                "CONV": {
                    "category": "feature",
                    "f_name": "Conv2D",
                    "filters": [8, 64, 8],
                },
                "GAP": {
                    "category": "pooling",
                    "f_name": "GlobalAveragePooling2D()",
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            registry = SearchSpaceRegistry.from_yaml(temp_path, validate=False)

            self.assertIn("CONV", registry.layer_names)
            self.assertIn("GAP", registry.layer_names)
            self.assertEqual(registry.get_start_layers(), ["CONV"])
        finally:
            os.unlink(temp_path)

    def test_parse_param_spec_discrete(self):
        """Test parsing [start, stop, step] parameter format."""
        spec = SearchSpaceRegistry._parse_param_spec("filters", [8, 32, 8])

        self.assertEqual(spec.param_type, "discrete")
        self.assertEqual(spec.values, [8, 16, 24, 32])

    def test_parse_param_spec_categorical(self):
        """Test parsing categorical parameter format."""
        spec = SearchSpaceRegistry._parse_param_spec(
            "activation", ["relu", "sigmoid", "tanh"]
        )

        self.assertEqual(spec.param_type, "categorical")
        self.assertEqual(spec.values, ["relu", "sigmoid", "tanh"])


class TestRandomChromosomeGeneration(unittest.TestCase):
    """Test random chromosome generation."""

    def setUp(self):
        """Set up a search space for random generation tests."""
        self.layers = {
            "CONV": LayerSpec(
                name="CONV",
                f_name="Conv2D",
                category="feature",
                params={
                    "filters": ParameterSpec(
                        name="filters",
                        param_type="discrete",
                        values=[8, 16, 32],
                    ),
                },
                successors=["CONV", "GAP"],
            ),
            "GAP": LayerSpec(
                name="GAP",
                f_name="GlobalAveragePooling2D",
                category="global_pooling",
                params={},
                successors=["D"],
            ),
            "D": LayerSpec(
                name="D",
                f_name="Dense",
                category="classification",
                params={
                    "units": ParameterSpec(
                        name="units",
                        param_type="discrete",
                        values=[32, 64],
                    ),
                },
                successors=["D"],
                terminal=True,
            ),
        }

        self.registry = SearchSpaceRegistry(
            layers=self.layers,
            categories={
                "feature": {},
                "global_pooling": {},
                "classification": {"terminal": True},
            },
            start_layers=["CONV"],
            max_layers=12,
        )

    def test_random_chromosome_is_valid(self):
        """Test that randomly generated chromosomes are valid."""
        for _ in range(10):
            chromosome = self.registry.create_random_chromosome(
                min_feature_layers=2,
                max_feature_layers=4,
                min_classification_layers=1,
                max_classification_layers=2,
            )

            self.assertGreater(len(chromosome), 0)

            # Check first layer is valid start
            self.assertIn(chromosome[0]["layer"], self.registry.get_start_layers())

            # Check all layers have required fields
            for gene in chromosome:
                self.assertIn("layer", gene)
                self.assertIn("f_name", gene)

    def test_random_chromosome_has_valid_params(self):
        """Test that random chromosomes have valid parameter values."""
        chromosome = self.registry.create_random_chromosome()

        for gene in chromosome:
            layer_name = gene["layer"]
            layer_spec = self.registry.layers[layer_name]

            for param_name, param_spec in layer_spec.params.items():
                if param_name in gene:
                    value = gene[param_name]
                    if param_spec.param_type in ("discrete", "categorical"):
                        self.assertIn(
                            value,
                            param_spec.values,
                            f"Invalid value {value} for param {param_name}",
                        )


class TestValidation(unittest.TestCase):
    """Test validation against LayerRegistry."""

    def setUp(self):
        """Clear and set up the layer registry."""
        LayerRegistry.clear()

        @LayerRegistry.register()
        def ValidLayer():
            pass

    def test_validation_passes_for_registered_layers(self):
        """Test that validation passes when all layers are registered."""
        layers = {
            "VALID": LayerSpec(
                name="VALID",
                f_name="ValidLayer",
                category="test",
                params={},
            ),
        }

        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["VALID"],
            max_layers=4,
        )

        # Should not raise
        registry.validate()

    def test_validation_fails_for_unregistered_layers(self):
        """Test that validation fails for unregistered layers."""
        layers = {
            "INVALID": LayerSpec(
                name="INVALID",
                f_name="NonExistentLayer",
                category="test",
                params={},
            ),
        }

        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["INVALID"],
            max_layers=4,
        )

        with self.assertRaises(SearchSpaceValidationError) as ctx:
            registry.validate()

        self.assertEqual(len(ctx.exception.errors), 1)
        self.assertEqual(ctx.exception.errors[0]["layer"], "INVALID")


if __name__ == "__main__":
    unittest.main()
