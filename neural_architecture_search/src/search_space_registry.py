"""
Search Space Registry for Neural Architecture Search.

This module provides the SearchSpaceRegistry class that handles:
- Loading and validating search space configurations
- Encoding chromosomes to vectors and decoding vectors back to chromosomes
- Enforcing connectivity rules during decoding
- Auto-generating encoding schemas from search space definitions

The registry supports a unified YAML schema where layer definitions and
connectivity rules are merged for easier maintenance.

Usage:
    # Load from YAML
    registry = SearchSpaceRegistry.from_yaml("conf/search_space/cifar10.yaml")

    # Encode a chromosome to vector
    vector = registry.encode(chromosome)

    # Decode back with rule enforcement
    chromosome = registry.decode(vector, enforce_rules=True)

    # Get schema information
    print(registry.vector_size)
    print(registry.layer_names)
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import copy

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from neural_architecture_search.src.layer_registry import (
    LayerRegistry,
    LayerNotFoundError,
)


class SearchSpaceValidationError(Exception):
    """Raised when search space configuration is invalid."""

    def __init__(self, errors: List[Dict[str, Any]], available_layers: List[str]):
        self.errors = errors
        self.available_layers = available_layers
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Search space validation failed:\n"]

        for err in self.errors:
            lines.append(f"  [x] Layer '{err['layer']}' uses f_name '{err['f_name']}' "
                        f"which is not registered.")
            if err.get("suggestion"):
                lines.append(f"      Did you mean: '{err['suggestion']}'?")

        lines.append(f"\nRegistered layers: {', '.join(self.available_layers[:15])}...")
        lines.append("\nTo register a custom layer, use @LayerRegistry.register()")

        return "\n".join(lines)


@dataclass
class ParameterSpec:
    """Specification for a layer parameter."""

    name: str
    param_type: str  # 'discrete', 'continuous', 'categorical'
    values: Optional[List[Any]] = None  # For discrete/categorical
    min_val: Optional[float] = None  # For continuous
    max_val: Optional[float] = None  # For continuous
    step: Optional[float] = None  # For discrete numeric

    @property
    def num_values(self) -> int:
        """Number of possible values for this parameter."""
        if self.param_type == "categorical":
            return len(self.values)
        elif self.param_type == "discrete":
            return len(self.values)
        else:  # continuous
            return 1  # Single normalized value

    @property
    def vector_size(self) -> int:
        """Size in the encoded vector (one-hot for categorical, 1 for others)."""
        if self.param_type == "categorical":
            return len(self.values)
        else:
            return 1


@dataclass
class LayerSpec:
    """Specification for a layer in the search space."""

    name: str  # Layer identifier (e.g., 'C_2D_BLOCK')
    f_name: str  # Function/class name (e.g., 'get_conv2d_block')
    category: str  # Category (e.g., 'feature_extraction')
    params: Dict[str, ParameterSpec] = field(default_factory=dict)
    successors: Optional[List[str]] = None  # Direct successors (overrides category)
    terminal: bool = False  # Can this layer end the architecture?

    @property
    def vector_size(self) -> int:
        """Total vector size needed to encode this layer's parameters."""
        return sum(p.vector_size for p in self.params.values())


@dataclass
class EncodingSchema:
    """Schema describing how to encode/decode architectures."""

    layer_names: List[str]  # Ordered list of layer names
    layer_to_idx: Dict[str, int]  # Map layer name to index
    all_params: Dict[str, ParameterSpec]  # Union of all parameters
    max_layers: int  # Maximum architecture depth
    layer_vector_size: int  # Size for layer type encoding (one-hot)
    param_vector_size: int  # Size for all parameters
    slot_size: int  # Total size per layer slot

    @property
    def total_vector_size(self) -> int:
        """Total vector size for encoding an architecture."""
        return self.slot_size * self.max_layers


class SearchSpaceRegistry:
    """
    Registry for search space configuration with encoding/decoding support.

    This class manages the search space definition and provides methods to:
    - Validate that all layers are implemented
    - Encode chromosomes to fixed-length vectors
    - Decode vectors back to chromosomes with rule enforcement
    - Create random valid chromosomes
    """

    def __init__(
        self,
        layers: Dict[str, LayerSpec],
        categories: Dict[str, Dict[str, Any]],
        start_layers: List[str],
        max_layers: int = 16,
        layer_registry: Optional[LayerRegistry] = None,
    ):
        """
        Initialize the SearchSpaceRegistry.

        Args:
            layers: Dict mapping layer names to LayerSpec objects
            categories: Dict mapping category names to their properties
            start_layers: List of valid starting layer names
            max_layers: Maximum number of layers in an architecture
            layer_registry: Optional LayerRegistry for validation
        """
        self.layers = layers
        self.categories = categories
        self.start_layers = start_layers
        self.max_layers = max_layers
        self._layer_registry = layer_registry

        # Build derived structures
        self._build_successor_map()
        self._build_encoding_schema()

    def _build_successor_map(self) -> None:
        """Build the map of valid successors for each layer."""
        self._successors: Dict[str, List[str]] = {}

        for layer_name, layer_spec in self.layers.items():
            if layer_spec.successors is not None:
                # Layer has explicit successors
                resolved = self._resolve_successors(layer_spec.successors)
            else:
                # Inherit from category
                category = self.categories.get(layer_spec.category, {})
                category_successors = category.get("successors", [])
                resolved = self._resolve_successors(category_successors)

            self._successors[layer_name] = resolved

    def _resolve_successors(self, successor_list: List[str]) -> List[str]:
        """Resolve successor references (can be layer names or category names)."""
        resolved = []
        for item in successor_list:
            if item in self.categories:
                # It's a category - expand to all layers in that category
                for layer_name, layer_spec in self.layers.items():
                    if layer_spec.category == item:
                        resolved.append(layer_name)
            else:
                # It's a specific layer name
                if item in self.layers:
                    resolved.append(item)
        return resolved

    def _build_encoding_schema(self) -> None:
        """Build the encoding schema from layer specifications."""
        layer_names = sorted(self.layers.keys())
        layer_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        # Collect all unique parameters across all layers
        all_params: Dict[str, ParameterSpec] = {}
        for layer_spec in self.layers.values():
            for param_name, param_spec in layer_spec.params.items():
                if param_name not in all_params:
                    all_params[param_name] = param_spec
                else:
                    # Merge parameter specs (expand values if needed)
                    existing = all_params[param_name]
                    if existing.param_type == "categorical" and param_spec.param_type == "categorical":
                        # Merge categorical values
                        merged_values = list(set(existing.values) | set(param_spec.values))
                        all_params[param_name] = ParameterSpec(
                            name=param_name,
                            param_type="categorical",
                            values=sorted(merged_values),
                        )

        # Calculate vector sizes
        layer_vector_size = len(layer_names) + 1  # +1 for "empty/padding" slot
        param_vector_size = sum(p.vector_size for p in all_params.values())
        slot_size = layer_vector_size + param_vector_size

        self.schema = EncodingSchema(
            layer_names=layer_names,
            layer_to_idx=layer_to_idx,
            all_params=all_params,
            max_layers=self.max_layers,
            layer_vector_size=layer_vector_size,
            param_vector_size=param_vector_size,
            slot_size=slot_size,
        )

    @property
    def vector_size(self) -> int:
        """Total size of encoded vectors."""
        return self.schema.total_vector_size

    @property
    def layer_names(self) -> List[str]:
        """List of all layer names in the search space."""
        return self.schema.layer_names

    def get_successors(self, layer_name: str) -> List[str]:
        """Get valid successor layers for a given layer."""
        return self._successors.get(layer_name, [])

    def get_start_layers(self) -> List[str]:
        """Get valid starting layers."""
        return self.start_layers

    def is_terminal(self, layer_name: str) -> bool:
        """Check if a layer can be terminal (end of architecture)."""
        layer_spec = self.layers.get(layer_name)
        if layer_spec is None:
            return False
        if layer_spec.terminal:
            return True
        # Also check category
        category = self.categories.get(layer_spec.category, {})
        return category.get("terminal", False)

    # =========================================================================
    # Encoding
    # =========================================================================

    def encode(self, chromosome: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode a chromosome (list of gene dicts) to a fixed-length vector.

        Args:
            chromosome: List of gene dictionaries

        Returns:
            Numpy array of shape (vector_size,)
        """
        vector = np.zeros(self.schema.total_vector_size, dtype=np.float32)

        for slot_idx, gene in enumerate(chromosome):
            if slot_idx >= self.max_layers:
                break

            slot_start = slot_idx * self.schema.slot_size
            self._encode_gene(gene, vector, slot_start)

        return vector

    def _encode_gene(
        self, gene: Dict[str, Any], vector: np.ndarray, offset: int
    ) -> None:
        """Encode a single gene into the vector at the given offset."""
        layer_name = gene.get("layer")
        if layer_name is None:
            return

        # Encode layer type (one-hot, with index 0 reserved for empty)
        if layer_name in self.schema.layer_to_idx:
            layer_idx = self.schema.layer_to_idx[layer_name] + 1  # +1 for empty slot
            vector[offset + layer_idx] = 1.0

        # Encode parameters
        param_offset = offset + self.schema.layer_vector_size
        for param_name, param_spec in self.schema.all_params.items():
            if param_name in gene:
                value = gene[param_name]
                self._encode_param(value, param_spec, vector, param_offset)
            param_offset += param_spec.vector_size

    def _encode_param(
        self,
        value: Any,
        param_spec: ParameterSpec,
        vector: np.ndarray,
        offset: int,
    ) -> None:
        """Encode a parameter value into the vector."""
        if param_spec.param_type == "categorical":
            # One-hot encoding
            if value in param_spec.values:
                idx = param_spec.values.index(value)
                vector[offset + idx] = 1.0
        elif param_spec.param_type == "discrete":
            # Normalize to [0, 1]
            if value in param_spec.values:
                idx = param_spec.values.index(value)
                vector[offset] = idx / max(len(param_spec.values) - 1, 1)
            else:
                # Find closest value
                closest_idx = min(
                    range(len(param_spec.values)),
                    key=lambda i: abs(param_spec.values[i] - value),
                )
                vector[offset] = closest_idx / max(len(param_spec.values) - 1, 1)
        else:  # continuous
            # Normalize to [0, 1]
            if param_spec.min_val is not None and param_spec.max_val is not None:
                range_val = param_spec.max_val - param_spec.min_val
                if range_val > 0:
                    vector[offset] = (value - param_spec.min_val) / range_val

    # =========================================================================
    # Decoding
    # =========================================================================

    def decode(
        self,
        vector: np.ndarray,
        enforce_rules: bool = True,
        min_layers: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Decode a vector back to a chromosome.

        Args:
            vector: Numpy array of shape (vector_size,)
            enforce_rules: If True, mask invalid layer choices based on rules
            min_layers: Minimum number of layers in the decoded architecture

        Returns:
            List of gene dictionaries
        """
        chromosome = []
        previous_layer = None

        for slot_idx in range(self.max_layers):
            slot_start = slot_idx * self.schema.slot_size
            slot_vector = vector[slot_start : slot_start + self.schema.slot_size]

            # Check if this is an empty slot
            if self._is_empty_slot(slot_vector):
                if len(chromosome) >= min_layers:
                    break
                continue

            # Decode layer type
            if enforce_rules:
                if previous_layer is None:
                    valid_layers = self.start_layers
                else:
                    valid_layers = self.get_successors(previous_layer)
                layer_name = self._decode_layer_type(slot_vector, valid_layers)
            else:
                layer_name = self._decode_layer_type(slot_vector)

            if layer_name is None:
                continue

            # Decode parameters
            gene = self._decode_gene(slot_vector, layer_name)
            chromosome.append(gene)
            previous_layer = layer_name

        return chromosome

    def _is_empty_slot(self, slot_vector: np.ndarray) -> bool:
        """Check if a slot vector represents an empty/padding slot."""
        layer_scores = slot_vector[: self.schema.layer_vector_size]
        # Empty if the "empty" index (0) has the highest score or all zeros
        return layer_scores[0] > np.max(layer_scores[1:]) or np.max(layer_scores) < 0.1

    def _decode_layer_type(
        self,
        slot_vector: np.ndarray,
        valid_layers: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Decode the layer type from a slot vector."""
        layer_scores = slot_vector[1 : self.schema.layer_vector_size].copy()

        if valid_layers is not None and len(valid_layers) > 0:
            # Mask invalid layers
            for idx, layer_name in enumerate(self.schema.layer_names):
                if layer_name not in valid_layers:
                    layer_scores[idx] = -np.inf

        if np.all(layer_scores == -np.inf):
            # No valid layers, pick the first valid one
            if valid_layers:
                return valid_layers[0]
            return None

        layer_idx = np.argmax(layer_scores)
        return self.schema.layer_names[layer_idx]

    def _decode_gene(
        self, slot_vector: np.ndarray, layer_name: str
    ) -> Dict[str, Any]:
        """Decode a full gene (layer + parameters) from a slot vector."""
        layer_spec = self.layers[layer_name]
        gene = {
            "layer": layer_name,
            "f_name": layer_spec.f_name,
        }

        # Decode parameters
        param_offset = self.schema.layer_vector_size
        for param_name, param_spec in self.schema.all_params.items():
            if param_name in layer_spec.params:
                value = self._decode_param(
                    slot_vector, param_offset, layer_spec.params[param_name]
                )
                gene[param_name] = value
            param_offset += param_spec.vector_size

        return gene

    def _decode_param(
        self,
        slot_vector: np.ndarray,
        offset: int,
        param_spec: ParameterSpec,
    ) -> Any:
        """Decode a parameter value from the vector."""
        if param_spec.param_type == "categorical":
            # One-hot decoding
            scores = slot_vector[offset : offset + param_spec.vector_size]
            idx = np.argmax(scores)
            return param_spec.values[idx]
        elif param_spec.param_type == "discrete":
            # Denormalize from [0, 1]
            normalized = slot_vector[offset]
            idx = int(round(normalized * (len(param_spec.values) - 1)))
            idx = max(0, min(idx, len(param_spec.values) - 1))
            return param_spec.values[idx]
        else:  # continuous
            # Denormalize from [0, 1]
            normalized = slot_vector[offset]
            return param_spec.min_val + normalized * (
                param_spec.max_val - param_spec.min_val
            )

    # =========================================================================
    # Random chromosome generation
    # =========================================================================

    def create_random_chromosome(
        self,
        min_feature_layers: int = 3,
        max_feature_layers: int = 8,
        min_classification_layers: int = 1,
        max_classification_layers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Create a random valid chromosome.

        Args:
            min_feature_layers: Minimum number of feature extraction layers
            max_feature_layers: Maximum number of feature extraction layers
            min_classification_layers: Minimum number of classification layers
            max_classification_layers: Maximum number of classification layers

        Returns:
            List of gene dictionaries representing a valid architecture
        """
        chromosome = []

        # Start with a valid starting layer
        layer_name = np.random.choice(self.start_layers)
        chromosome.append(self._create_random_gene(layer_name))

        # Add feature extraction layers
        num_feature_layers = np.random.randint(
            min_feature_layers, max_feature_layers + 1
        )
        for _ in range(num_feature_layers):
            successors = self.get_successors(layer_name)
            if not successors:
                break
            layer_name = np.random.choice(successors)
            chromosome.append(self._create_random_gene(layer_name))

        # Find a path to classification layers (through pooling)
        # Look for global pooling layer
        pooling_layers = [
            name
            for name, spec in self.layers.items()
            if spec.category in ("pooling", "global_pooling", "global_pooling_2D", "global_pooling_1D")
        ]

        if pooling_layers:
            # Find a valid pooling layer that can follow the current layer
            successors = self.get_successors(layer_name)
            valid_pooling = [p for p in pooling_layers if p in successors]
            if valid_pooling:
                layer_name = np.random.choice(valid_pooling)
                chromosome.append(self._create_random_gene(layer_name))

        # Add classification layers (Dense)
        num_classification_layers = np.random.randint(
            min_classification_layers, max_classification_layers + 1
        )
        for _ in range(num_classification_layers):
            successors = self.get_successors(layer_name)
            if not successors:
                break
            # Prefer Dense layers for classification
            dense_successors = [s for s in successors if "D" in s or "Dense" in s]
            if dense_successors:
                layer_name = np.random.choice(dense_successors)
            else:
                layer_name = np.random.choice(successors)
            chromosome.append(self._create_random_gene(layer_name))

        return chromosome

    def _create_random_gene(self, layer_name: str) -> Dict[str, Any]:
        """Create a gene with random valid parameters for a layer."""
        layer_spec = self.layers[layer_name]
        gene = {
            "layer": layer_name,
            "f_name": layer_spec.f_name,
        }

        for param_name, param_spec in layer_spec.params.items():
            gene[param_name] = self._sample_random_param(param_spec)

        return gene

    def _sample_random_param(self, param_spec: ParameterSpec) -> Any:
        """Sample a random value for a parameter."""
        if param_spec.param_type == "categorical":
            return np.random.choice(param_spec.values)
        elif param_spec.param_type == "discrete":
            return np.random.choice(param_spec.values)
        else:  # continuous
            return np.random.uniform(param_spec.min_val, param_spec.max_val)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, layer_registry: Optional[LayerRegistry] = None) -> None:
        """
        Validate that all layers in the search space are implemented.

        Args:
            layer_registry: LayerRegistry to validate against

        Raises:
            SearchSpaceValidationError: If validation fails
        """
        registry = layer_registry or self._layer_registry or LayerRegistry()
        errors = []

        for layer_name, layer_spec in self.layers.items():
            f_name = layer_spec.f_name.rstrip("()")

            if not registry.exists(f_name):
                from difflib import get_close_matches

                available = registry.list_available()
                suggestions = get_close_matches(f_name, available, n=1, cutoff=0.4)

                errors.append(
                    {
                        "layer": layer_name,
                        "f_name": layer_spec.f_name,
                        "suggestion": suggestions[0] if suggestions else None,
                    }
                )

        if errors:
            raise SearchSpaceValidationError(
                errors, (layer_registry or LayerRegistry()).list_available()
            )

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_schema(self, path: Union[str, Path]) -> None:
        """Save the encoding schema to a JSON file."""
        schema_dict = {
            "layer_names": self.schema.layer_names,
            "max_layers": self.schema.max_layers,
            "slot_size": self.schema.slot_size,
            "total_vector_size": self.schema.total_vector_size,
            "params": {
                name: {
                    "type": spec.param_type,
                    "values": spec.values,
                    "min": spec.min_val,
                    "max": spec.max_val,
                }
                for name, spec in self.schema.all_params.items()
            },
        }
        with open(path, "w") as f:
            json.dump(schema_dict, f, indent=2)

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        layer_registry: Optional[LayerRegistry] = None,
        validate: bool = True,
        max_layers: int = 16,
    ) -> "SearchSpaceRegistry":
        """
        Create a SearchSpaceRegistry from a YAML configuration file.

        Supports both the new unified format and the legacy format.

        Args:
            path: Path to the YAML file
            layer_registry: Optional LayerRegistry for validation
            validate: Whether to validate layers against the registry
            max_layers: Maximum architecture depth

        Returns:
            SearchSpaceRegistry instance
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls.from_dict(
            config,
            layer_registry=layer_registry,
            validate=validate,
            max_layers=max_layers,
        )

    @classmethod
    def from_omegaconf(
        cls,
        cfg: DictConfig,
        layer_registry: Optional[LayerRegistry] = None,
        validate: bool = True,
        max_layers: int = 16,
    ) -> "SearchSpaceRegistry":
        """
        Create a SearchSpaceRegistry from an OmegaConf DictConfig.

        Args:
            cfg: OmegaConf configuration (typically cfg.search_space)
            layer_registry: Optional LayerRegistry for validation
            validate: Whether to validate layers against the registry
            max_layers: Maximum architecture depth

        Returns:
            SearchSpaceRegistry instance
        """
        config = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_dict(
            config,
            layer_registry=layer_registry,
            validate=validate,
            max_layers=max_layers,
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        layer_registry: Optional[LayerRegistry] = None,
        validate: bool = True,
        max_layers: int = 16,
    ) -> "SearchSpaceRegistry":
        """
        Create a SearchSpaceRegistry from a configuration dict.

        Supports both unified format (categories + layers) and legacy format
        (gene_pool + rule_set).

        Args:
            config: Configuration dictionary
            layer_registry: Optional LayerRegistry for validation
            validate: Whether to validate layers against the registry
            max_layers: Maximum architecture depth

        Returns:
            SearchSpaceRegistry instance
        """
        # Detect format and parse accordingly
        if "layers" in config and "categories" in config:
            # New unified format
            layers, categories, start_layers = cls._parse_unified_format(config)
        elif "gene_pool" in config:
            # Legacy format
            layers, categories, start_layers = cls._parse_legacy_format(config)
        else:
            raise ValueError(
                "Invalid config format. Expected 'layers' + 'categories' "
                "or 'gene_pool' + 'rule_set'."
            )

        registry = cls(
            layers=layers,
            categories=categories,
            start_layers=start_layers,
            max_layers=max_layers,
            layer_registry=layer_registry,
        )

        if validate:
            registry.validate(layer_registry)

        return registry

    @classmethod
    def _parse_unified_format(
        cls, config: Dict[str, Any]
    ) -> Tuple[Dict[str, LayerSpec], Dict[str, Dict], List[str]]:
        """Parse the new unified configuration format."""
        categories = config.get("categories", {})
        layers_config = config.get("layers", {})
        start_layers = config.get("start", [])

        layers = {}
        for layer_name, layer_def in layers_config.items():
            params = {}
            for key, value in layer_def.items():
                if key in ("category", "f_name", "successors", "terminal"):
                    continue
                params[key] = cls._parse_param_spec(key, value)

            layers[layer_name] = LayerSpec(
                name=layer_name,
                f_name=layer_def.get("f_name", layer_name),
                category=layer_def.get("category", "unknown"),
                params=params,
                successors=layer_def.get("successors"),
                terminal=layer_def.get("terminal", False),
            )

        return layers, categories, start_layers

    @classmethod
    def _parse_legacy_format(
        cls, config: Dict[str, Any]
    ) -> Tuple[Dict[str, LayerSpec], Dict[str, Dict], List[str]]:
        """Parse the legacy gene_pool + rule_set format."""
        gene_pool = config.get("gene_pool", {})
        rule_set = config.get("rule_set", {})

        # Build category mapping from gene_pool structure
        categories = {}
        for category_name in gene_pool.keys():
            categories[category_name] = {"successors": []}

        # Parse layers from gene_pool
        layers = {}
        for category_name, layer_list in gene_pool.items():
            for layer_def in layer_list:
                layer_name = layer_def.get("layer")
                if not layer_name:
                    continue

                params = {}
                for key, value in layer_def.items():
                    if key in ("layer", "f_name"):
                        continue
                    params[key] = cls._parse_param_spec(key, value)

                # Get successors from rule_set
                rule_entry = rule_set.get(layer_name, {})
                successors = rule_entry.get("rule")

                layers[layer_name] = LayerSpec(
                    name=layer_name,
                    f_name=layer_def.get("f_name", layer_name),
                    category=category_name,
                    params=params,
                    successors=successors,
                    terminal=False,
                )

        # Get start layers from rule_set
        start_entry = rule_set.get("Start", {})
        start_layers = start_entry.get("rule", [])

        return layers, categories, start_layers

    @classmethod
    def _parse_param_spec(cls, name: str, value: Any) -> ParameterSpec:
        """Parse a parameter specification from config value."""
        if isinstance(value, list):
            if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
                # [start, stop, step] format for discrete numeric
                start, stop, step = value
                values = list(np.arange(start, stop + step, step))
                # Convert to int if original values were int
                if all(isinstance(v, int) for v in value):
                    values = [int(v) for v in values]
                return ParameterSpec(
                    name=name,
                    param_type="discrete",
                    values=values,
                    min_val=float(start),
                    max_val=float(stop),
                    step=float(step),
                )
            else:
                # Categorical list
                return ParameterSpec(
                    name=name,
                    param_type="categorical",
                    values=list(value),
                )
        else:
            # Single value - treat as categorical with one option
            return ParameterSpec(
                name=name,
                param_type="categorical",
                values=[value],
            )
