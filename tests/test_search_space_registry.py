"""Tests for the SearchSpaceRegistry class."""

import unittest
import tempfile
import os
import json

import numpy as np
import yaml
from omegaconf import OmegaConf, DictConfig

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

    def test_continuous_param_num_values(self):
        """Test that continuous parameters report num_values == 1."""
        spec = ParameterSpec(
            name="x",
            param_type="continuous",
            min_val=0.0,
            max_val=1.0,
        )
        self.assertEqual(spec.num_values, 1)

    def test_single_value_categorical(self):
        """Test categorical parameter with a single value."""
        spec = ParameterSpec(
            name="choice",
            param_type="categorical",
            values=["only"],
        )
        self.assertEqual(spec.num_values, 1)
        self.assertEqual(spec.vector_size, 1)

    def test_discrete_param_vector_size_is_one(self):
        """Test that discrete parameters always have vector_size == 1."""
        spec = ParameterSpec(
            name="filters",
            param_type="discrete",
            values=[8, 16, 32, 64],
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

    def test_is_terminal_via_category(self):
        """Test is_terminal returns True when the category has terminal: True."""
        # Create a new layer that is not explicitly terminal but whose category is
        layers = dict(self.layers)
        layers["SOFTMAX"] = LayerSpec(
            name="SOFTMAX",
            f_name="Softmax",
            category="classification",
            params={},
            terminal=False,  # Not explicitly terminal
        )
        registry = SearchSpaceRegistry(
            layers=layers,
            categories=self.categories,
            start_layers=["CONV"],
            max_layers=8,
        )
        # classification category has terminal: True
        self.assertTrue(registry.is_terminal("SOFTMAX"))

    def test_is_terminal_unknown_layer(self):
        """Test is_terminal returns False for a nonexistent layer."""
        self.assertFalse(self.registry.is_terminal("NONEXISTENT"))

    def test_get_successors_nonexistent_layer(self):
        """Test get_successors returns [] for a nonexistent layer."""
        self.assertEqual(self.registry.get_successors("NONEXISTENT"), [])

    def test_get_successors_category_resolution(self):
        """Test that successors defined at category level are resolved correctly."""
        layers = {
            "CONV_A": LayerSpec(
                name="CONV_A",
                f_name="Conv2D",
                category="feature",
                params={},
                # No explicit successors — should inherit from category
            ),
            "POOL_A": LayerSpec(
                name="POOL_A",
                f_name="MaxPooling2D",
                category="pooling",
                params={},
            ),
        }
        categories = {
            "feature": {"successors": ["pooling"]},
            "pooling": {},
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories=categories,
            start_layers=["CONV_A"],
            max_layers=4,
        )
        # CONV_A's category "feature" has successors=["pooling"]
        # "pooling" is a category, so it expands to all layers in "pooling" → ["POOL_A"]
        successors = registry.get_successors("CONV_A")
        self.assertIn("POOL_A", successors)

    def test_vector_size_exact_computation(self):
        """Test exact vector_size computation for the setUp registry."""
        # 3 layers (CONV, DENSE, POOL) sorted → layer_vector_size = 3 + 1 = 4
        # Params union: "activation" (categorical, 2 values → size 2),
        #               "filters" (discrete → size 1),
        #               "units" (discrete → size 1)
        # param_vector_size = 2 + 1 + 1 = 4
        # slot_size = 4 + 4 = 8
        # total = 8 * 8 (max_layers) = 64
        self.assertEqual(self.registry.vector_size, 64)




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

    def test_encode_empty_chromosome(self):
        """Test encoding an empty chromosome returns all zeros."""
        vector = self.registry.encode([])
        self.assertEqual(vector.shape, (self.registry.vector_size,))
        np.testing.assert_array_equal(vector, np.zeros(self.registry.vector_size, dtype=np.float32))

    def test_encode_truncates_at_max_layers(self):
        """Test that encoding truncates chromosomes longer than max_layers."""
        small_registry = SearchSpaceRegistry(
            layers=self.layers,
            categories={},
            start_layers=["CONV"],
            max_layers=4,
        )
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 32},
            {"layer": "CONV", "f_name": "Conv2D", "filters": 16},
            {"layer": "CONV", "f_name": "Conv2D", "filters": 8},
            {"layer": "CONV", "f_name": "Conv2D", "filters": 64},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
            {"layer": "DENSE", "f_name": "Dense", "units": 64},
        ]
        vector = small_registry.encode(chromosome)
        # Only first 4 slots should be populated
        slot_size = small_registry.schema.slot_size
        for slot_idx in range(4):
            slot = vector[slot_idx * slot_size : (slot_idx + 1) * slot_size]
            self.assertTrue(np.any(slot != 0), f"Slot {slot_idx} should be populated")
        # No slots beyond 4 (vector only has 4 slots)
        self.assertEqual(vector.shape[0], slot_size * 4)

    def test_encode_gene_missing_layer_key(self):
        """Test encoding a gene without the 'layer' key produces an empty slot."""
        vector = self.registry.encode([{"f_name": "Conv2D", "filters": 32}])
        slot_size = self.registry.schema.slot_size
        slot = vector[:slot_size]
        np.testing.assert_array_equal(slot, np.zeros(slot_size, dtype=np.float32))

    def test_encode_unknown_layer(self):
        """Test encoding a gene with an unknown layer name leaves layer one-hot as zeros."""
        vector = self.registry.encode([{"layer": "UNKNOWN", "f_name": "X"}])
        slot_size = self.registry.schema.slot_size
        layer_vector_size = self.registry.schema.layer_vector_size
        layer_one_hot = vector[:layer_vector_size]
        np.testing.assert_array_equal(layer_one_hot, np.zeros(layer_vector_size, dtype=np.float32))

    def test_decode_all_zeros_vector(self):
        """Test decoding an all-zeros vector returns an empty chromosome."""
        vector = np.zeros(self.registry.vector_size, dtype=np.float32)
        chromosome = self.registry.decode(vector, enforce_rules=False)
        self.assertEqual(chromosome, [])

    def test_roundtrip_preserves_parameter_values(self):
        """Test that encode/decode roundtrip preserves parameter values."""
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 32},
        ]
        vector = self.registry.encode(chromosome)
        decoded = self.registry.decode(vector, enforce_rules=False)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["filters"], 32)

    def test_is_empty_slot_clear_empty(self):
        """Test _is_empty_slot returns True when index 0 is highest."""
        slot_vector = np.zeros(self.registry.schema.slot_size, dtype=np.float32)
        slot_vector[0] = 1.0  # "empty" index has highest score
        self.assertTrue(self.registry._is_empty_slot(slot_vector))

    def test_is_empty_slot_low_activation(self):
        """Test _is_empty_slot returns True when all activations are very low."""
        slot_vector = np.full(self.registry.schema.slot_size, 0.05, dtype=np.float32)
        self.assertTrue(self.registry._is_empty_slot(slot_vector))

    def test_is_empty_slot_populated(self):
        """Test _is_empty_slot returns False when a non-empty layer has highest score."""
        slot_vector = np.zeros(self.registry.schema.slot_size, dtype=np.float32)
        slot_vector[0] = 0.0   # empty index
        slot_vector[1] = 1.0   # first real layer has highest score
        self.assertFalse(self.registry._is_empty_slot(slot_vector))

    def test_early_stop_threshold_prevents_early_termination(self):
        """Test that early_stop_threshold allows all populated slots to be decoded."""
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 8},
            {"layer": "CONV", "f_name": "Conv2D", "filters": 16},
            {"layer": "CONV", "f_name": "Conv2D", "filters": 32},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
            {"layer": "DENSE", "f_name": "Dense", "units": 64},
        ]
        vector = self.registry.encode(chromosome)
        decoded = self.registry.decode(vector, enforce_rules=False, early_stop_threshold=3)
        self.assertEqual(len(decoded), 5)

    def test_early_stop_threshold_does_not_guarantee_minimum(self):
        """Test that early_stop_threshold does NOT guarantee a minimum number of layers."""
        # Only populate 2 slots, rest are zeros
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D", "filters": 8},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
        ]
        vector = self.registry.encode(chromosome)
        # Decode with a high early_stop_threshold
        decoded = self.registry.decode(vector, enforce_rules=False, early_stop_threshold=5)
        # Result should have at most 2 layers (only 2 populated slots)
        self.assertLessEqual(len(decoded), 2)

    def test_decode_with_no_valid_successors(self):
        """Test decoding handles layers with empty successor lists gracefully."""
        layers = {
            "CONV": LayerSpec(
                name="CONV",
                f_name="Conv2D",
                category="feature",
                params={},
                successors=["POOL"],
            ),
            "POOL": LayerSpec(
                name="POOL",
                f_name="GlobalAveragePooling2D",
                category="pooling",
                params={},
                successors=[],  # No valid successors
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["CONV"],
            max_layers=4,
        )
        chromosome = [
            {"layer": "CONV", "f_name": "Conv2D"},
            {"layer": "POOL", "f_name": "GlobalAveragePooling2D"},
        ]
        vector = registry.encode(chromosome)
        decoded = registry.decode(vector, enforce_rules=True)
        # Should contain at least CONV and POOL
        layer_names = [g["layer"] for g in decoded]
        self.assertIn("CONV", layer_names)
        self.assertIn("POOL", layer_names)

    def test_decode_layer_type_all_masked_fallback(self):
        """Test _decode_layer_type falls back to first valid layer when all are masked."""
        slot_vector = np.zeros(self.registry.schema.slot_size, dtype=np.float32)
        # Set CONV (index 0 in sorted layer_names) as the argmax
        # layer_names are sorted: ["CONV", "DENSE", "POOL"]
        # In the one-hot, index 0 is empty, index 1 is CONV, index 2 is DENSE, index 3 is POOL
        slot_vector[1] = 1.0  # CONV is argmax
        # But valid_layers only contains DENSE (which is NOT the argmax)
        result = self.registry._decode_layer_type(slot_vector, valid_layers=["DENSE"])
        self.assertEqual(result, "DENSE")

    def test_encode_categorical_param(self):
        """Test encoding of categorical parameter uses one-hot encoding."""
        # Create a registry with a categorical parameter
        layers = {
            "LAYER_A": LayerSpec(
                name="LAYER_A",
                f_name="LayerA",
                category="test",
                params={
                    "activation": ParameterSpec(
                        name="activation",
                        param_type="categorical",
                        values=["relu", "sigmoid"],
                    ),
                },
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["LAYER_A"],
            max_layers=2,
        )
        chromosome = [{"layer": "LAYER_A", "f_name": "LayerA", "activation": "sigmoid"}]
        vector = registry.encode(chromosome)
        # layer_vector_size = 1 + 1 = 2 (1 layer + empty)
        # activation param starts at offset 2 in the slot
        # sigmoid is index 1 → one-hot [0, 1]
        layer_vec_size = registry.schema.layer_vector_size
        self.assertEqual(vector[layer_vec_size], 0.0)      # relu position
        self.assertEqual(vector[layer_vec_size + 1], 1.0)  # sigmoid position




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

    def test_from_omegaconf(self):
        """Test creating a registry from an OmegaConf DictConfig."""
        cfg = OmegaConf.create({
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
        })
        registry = SearchSpaceRegistry.from_omegaconf(cfg, validate=False)
        self.assertIn("CONV", registry.layer_names)
        self.assertIn("GAP", registry.layer_names)
        self.assertEqual(registry.get_start_layers(), ["CONV"])

    def test_from_dict_invalid_format(self):
        """Test from_dict raises ValueError for invalid config format."""
        with self.assertRaises(ValueError):
            SearchSpaceRegistry.from_dict({}, validate=False)

    def test_parse_param_spec_single_value(self):
        """Test _parse_param_spec with a single scalar value."""
        spec = SearchSpaceRegistry._parse_param_spec("val", 42)
        self.assertEqual(spec.param_type, "categorical")
        self.assertEqual(spec.values, [42])

    def test_parse_param_spec_float_range(self):
        """Test _parse_param_spec with a float [start, stop, step] range."""
        spec = SearchSpaceRegistry._parse_param_spec("rate", [0.0, 0.5, 0.1])
        self.assertEqual(spec.param_type, "discrete")
        self.assertEqual(len(spec.values), 6)

    def test_load_actual_unified_yaml(self):
        """Test loading the actual speech_commands_unified.yaml config file."""
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "search_space",
            "speech_commands_unified.yaml",
        )
        registry = SearchSpaceRegistry.from_yaml(yaml_path, validate=False)
        self.assertEqual(len(registry.layers), 14)
        self.assertEqual(registry.get_start_layers(), ["STFT_2D"])

    def test_load_actual_legacy_yaml(self):
        """Test loading the actual cifar10.yaml config file (legacy format)."""
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "search_space",
            "cifar10.yaml",
        )
        registry = SearchSpaceRegistry.from_yaml(yaml_path, validate=False)
        self.assertTrue(len(registry.layers) > 0)




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

    def test_random_with_no_pooling_layers(self):
        """Test random chromosome generation when no pooling layers exist."""
        layers = {
            "CONV": LayerSpec(
                name="CONV",
                f_name="Conv2D",
                category="feature",
                params={},
                successors=["CONV", "D"],
            ),
            "D": LayerSpec(
                name="D",
                f_name="Dense",
                category="classification",
                params={},
                successors=["D"],
                terminal=True,
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={
                "feature": {},
                "classification": {"terminal": True},
            },
            start_layers=["CONV"],
            max_layers=12,
        )
        chromosome = registry.create_random_chromosome()
        self.assertGreater(len(chromosome), 0)
        self.assertEqual(chromosome[0]["layer"], "CONV")

    def test_random_with_custom_layer_counts(self):
        """Test random chromosome with min/max feature and classification counts."""
        chromosome = self.registry.create_random_chromosome(
            min_feature_layers=1,
            max_feature_layers=1,
            min_classification_layers=1,
            max_classification_layers=1,
        )
        # start layer + 1 feature + optional pooling + 1 classification = 3 or 4
        self.assertGreaterEqual(len(chromosome), 3)
        self.assertLessEqual(len(chromosome), 4)

    def test_random_determinism_with_seed(self):
        """Test that random chromosome generation is deterministic with a fixed seed."""
        np.random.seed(42)
        chr1 = self.registry.create_random_chromosome()
        np.random.seed(42)
        chr2 = self.registry.create_random_chromosome()
        self.assertEqual(chr1, chr2)




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

    def test_save_schema_produces_valid_json(self):
        """Test that save_schema produces a valid JSON file with expected keys."""
        layers = {
            "VALID": LayerSpec(
                name="VALID",
                f_name="ValidLayer",
                category="test",
                params={
                    "units": ParameterSpec(
                        name="units",
                        param_type="discrete",
                        values=[32, 64],
                    ),
                },
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["VALID"],
            max_layers=4,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            registry.save_schema(temp_path)
            with open(temp_path, "r") as f:
                schema = json.load(f)
            self.assertIn("layer_names", schema)
            self.assertIn("max_layers", schema)
            self.assertIn("slot_size", schema)
            self.assertIn("total_vector_size", schema)
            self.assertIn("params", schema)
        finally:
            os.unlink(temp_path)

    def test_validation_with_custom_layer_registry(self):
        """Test validation passes when using the default layer registry."""
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
        # Should not raise — ValidLayer is registered in setUp
        registry.validate(layer_registry=None)

    def test_validation_error_message_formatting(self):
        """Test that SearchSpaceValidationError message contains layer and f_name."""
        layers = {
            "BAD_LAYER": LayerSpec(
                name="BAD_LAYER",
                f_name="NonExistentFunc",
                category="test",
                params={},
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["BAD_LAYER"],
            max_layers=4,
        )
        with self.assertRaises(SearchSpaceValidationError) as ctx:
            registry.validate()
        error_msg = str(ctx.exception)
        self.assertIn("BAD_LAYER", error_msg)
        self.assertIn("NonExistentFunc", error_msg)

    def test_validation_error_with_similar_name(self):
        """Test that validation error suggests a similar registered name."""
        # Register "Dense" layer
        @LayerRegistry.register(name="Dense")
        def Dense():
            pass

        layers = {
            "MY_LAYER": LayerSpec(
                name="MY_LAYER",
                f_name="Denes",  # Typo of "Dense"
                category="test",
                params={},
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["MY_LAYER"],
            max_layers=4,
        )
        with self.assertRaises(SearchSpaceValidationError) as ctx:
            registry.validate()
        error_msg = str(ctx.exception)
        self.assertIn("Dense", error_msg)






class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Clear the layer registry before each test."""
        LayerRegistry.clear()

    def test_empty_search_space(self):
        """Test a search space with no layers."""
        registry = SearchSpaceRegistry(
            layers={},
            categories={},
            start_layers=[],
            max_layers=4,
        )
        self.assertIsNotNone(registry.vector_size)
        self.assertEqual(registry.layer_names, [])

    def test_single_layer_search_space(self):
        """Test a search space with a single self-referencing layer."""
        layers = {
            "ONLY": LayerSpec(
                name="ONLY",
                f_name="OnlyLayer",
                category="all",
                params={
                    "size": ParameterSpec(
                        name="size",
                        param_type="discrete",
                        values=[1, 2, 3],
                    ),
                },
                successors=["ONLY"],
                terminal=True,
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["ONLY"],
            max_layers=4,
        )
        chromosome = [{"layer": "ONLY", "f_name": "OnlyLayer", "size": 2}]
        vector = registry.encode(chromosome)
        decoded = registry.decode(vector, enforce_rules=True)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["layer"], "ONLY")
        self.assertEqual(decoded[0]["size"], 2)

    def test_very_large_max_layers(self):
        """Test that vector_size is computable for very large max_layers."""
        layers = {
            "A": LayerSpec(
                name="A",
                f_name="LayerA",
                category="test",
                params={},
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["A"],
            max_layers=1000,
        )
        slot_size = registry.schema.slot_size
        self.assertEqual(registry.vector_size, slot_size * 1000)

    def test_all_layers_are_terminal(self):
        """Test a search space where every layer is terminal."""
        layers = {
            "A": LayerSpec(
                name="A", f_name="LayerA", category="c",
                params={}, terminal=True, successors=["B", "C"],
            ),
            "B": LayerSpec(
                name="B", f_name="LayerB", category="c",
                params={}, terminal=True, successors=["A", "C"],
            ),
            "C": LayerSpec(
                name="C", f_name="LayerC", category="c",
                params={}, terminal=True, successors=["A", "B"],
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["A"],
            max_layers=4,
        )
        for name in ["A", "B", "C"]:
            self.assertTrue(registry.is_terminal(name))

    def test_resolve_successors_nonexistent_category(self):
        """Test that a successor referencing a nonexistent category resolves to empty."""
        layers = {
            "X": LayerSpec(
                name="X",
                f_name="LayerX",
                category="test",
                params={},
                successors=["nonexistent_category"],
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["X"],
            max_layers=4,
        )
        self.assertEqual(registry.get_successors("X"), [])

    def test_resolve_successors_nonexistent_layer(self):
        """Test that a successor referencing a nonexistent layer is silently ignored."""
        layers = {
            "Y": LayerSpec(
                name="Y",
                f_name="LayerY",
                category="test",
                params={},
                successors=["NONEXISTENT_LAYER"],
            ),
        }
        registry = SearchSpaceRegistry(
            layers=layers,
            categories={},
            start_layers=["Y"],
            max_layers=4,
        )
        self.assertEqual(registry.get_successors("Y"), [])



if __name__ == "__main__":
    unittest.main()
