"""
Example: PyMOO Integration for Multi-Objective Neural Architecture Search.

This module demonstrates how to use the SearchSpaceRegistry with PyMOO
for multi-objective optimization of neural architectures.

The example shows:
- Defining a NAS problem for PyMOO
- Custom sampling using the registry
- Custom repair operator for rule enforcement
- Running NSGA-II optimization

Requirements:
    pip install pymoo

Usage:
    python -m neural_architecture_search.examples.pymoo_integration
"""

import numpy as np

try:
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.core.sampling import Sampling
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("PyMOO not installed. Install with: pip install pymoo")

from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry
from neural_architecture_search.src.layer_registry import LayerRegistry


class NASProblem(Problem):
    """
    Multi-objective Neural Architecture Search problem for PyMOO.

    Objectives (all minimized):
    - Negative accuracy (to maximize accuracy)
    - Model size (parameters or memory)
    - Inference latency (or energy consumption)

    The problem uses vector representation from SearchSpaceRegistry,
    enabling gradient-free optimization over the architecture space.
    """

    def __init__(self, registry: SearchSpaceRegistry, evaluator=None):
        """
        Initialize the NAS problem.

        Args:
            registry: SearchSpaceRegistry instance
            evaluator: Optional callable that takes a chromosome and returns
                      (accuracy, size, latency). If None, uses dummy values.
        """
        self.registry = registry
        self.evaluator = evaluator or self._dummy_evaluator

        super().__init__(
            n_var=registry.vector_size,
            n_obj=3,  # accuracy, size, latency
            n_constr=0,  # constraints handled via repair
            xl=np.zeros(registry.vector_size),
            xu=np.ones(registry.vector_size),
        )

    def _dummy_evaluator(self, chromosome):
        """Dummy evaluator for demonstration purposes."""
        # In practice, this would:
        # 1. Translate chromosome to Keras model
        # 2. Train the model (or use weight sharing)
        # 3. Evaluate accuracy on validation set
        # 4. Measure model size and latency

        # Dummy values based on chromosome length
        n_layers = len(chromosome)
        accuracy = 0.5 + 0.1 * np.random.rand()  # Random accuracy
        size = n_layers * 1000 + np.random.randint(0, 5000)  # Params
        latency = n_layers * 0.5 + np.random.rand()  # ms

        return accuracy, size, latency

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate a population of architectures.

        Args:
            X: Population matrix of shape (pop_size, n_var)
            out: Output dictionary for objectives
        """
        objectives = []

        for vector in X:
            # Decode vector to chromosome
            chromosome = self.registry.decode(vector, enforce_rules=True)

            # Evaluate architecture
            accuracy, size, latency = self.evaluator(chromosome)

            # PyMOO minimizes, so negate accuracy
            objectives.append([-accuracy, size, latency])

        out["F"] = np.array(objectives)


class ArchitectureRepair(Repair):
    """
    Repair operator that enforces architecture validity.

    After crossover/mutation, vectors may represent invalid architectures.
    This repair operator decodes with rule enforcement, then re-encodes
    to get a valid vector.
    """

    def __init__(self, registry: SearchSpaceRegistry):
        """
        Initialize the repair operator.

        Args:
            registry: SearchSpaceRegistry instance
        """
        super().__init__()
        self.registry = registry

    def _do(self, problem, X, **kwargs):
        """
        Repair invalid individuals in the population.

        Args:
            problem: The optimization problem
            X: Population matrix

        Returns:
            Repaired population matrix
        """
        for i in range(len(X)):
            # Decode with rule enforcement
            chromosome = self.registry.decode(X[i], enforce_rules=True)

            # Re-encode to get valid vector
            if len(chromosome) > 0:
                X[i] = self.registry.encode(chromosome)

        return X


class ArchitectureSampling(Sampling):
    """
    Custom sampling that generates valid initial architectures.

    Uses the registry's random chromosome generation to create
    valid architectures, then encodes them to vectors.
    """

    def __init__(self, registry: SearchSpaceRegistry):
        """
        Initialize the sampling operator.

        Args:
            registry: SearchSpaceRegistry instance
        """
        super().__init__()
        self.registry = registry

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate initial population.

        Args:
            problem: The optimization problem
            n_samples: Number of samples to generate

        Returns:
            Population matrix of shape (n_samples, n_var)
        """
        X = np.zeros((n_samples, problem.n_var), dtype=np.float32)

        for i in range(n_samples):
            chromosome = self.registry.create_random_chromosome()
            X[i] = self.registry.encode(chromosome)

        return X


def run_example():
    """Run the PyMOO NAS example."""
    if not PYMOO_AVAILABLE:
        print("PyMOO is required for this example.")
        print("Install with: pip install pymoo")
        return

    print("Loading search space...")

    # Load the unified search space
    registry = SearchSpaceRegistry.from_yaml(
        "conf/search_space/speech_commands.yaml",
        validate=False,  # Skip validation for example
        max_layers=12,
    )

    print(f"Search space loaded:")
    print(f"  - {len(registry.layer_names)} layer types")
    print(f"  - Vector size: {registry.vector_size}")
    print(f"  - Max layers: {registry.max_layers}")
    print()

    # Create the optimization problem
    problem = NASProblem(registry)

    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=20,  # Small population for demo
        sampling=ArchitectureSampling(registry),
        repair=ArchitectureRepair(registry),
    )

    print("Running NSGA-II optimization...")
    print("(This is a demo with dummy fitness values)")
    print()

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        ("n_gen", 5),  # Few generations for demo
        verbose=True,
    )

    print()
    print("=" * 60)
    print("Optimization complete!")
    print(f"Found {len(result.X)} Pareto-optimal architectures")
    print()

    # Show some results
    print("Sample architectures from Pareto front:")
    print("-" * 60)

    for i, (vector, obj) in enumerate(zip(result.X[:3], result.F[:3])):
        chromosome = registry.decode(vector, enforce_rules=True)

        print(f"\nArchitecture {i + 1}:")
        print(f"  Accuracy: {-obj[0]:.3f}")
        print(f"  Size: {obj[1]:.0f} params")
        print(f"  Latency: {obj[2]:.2f} ms")
        print(f"  Layers ({len(chromosome)}):")
        for gene in chromosome:
            layer_name = gene["layer"]
            params = {k: v for k, v in gene.items() if k not in ("layer", "f_name")}
            if params:
                print(f"    - {layer_name}: {params}")
            else:
                print(f"    - {layer_name}")


def demonstrate_encoding():
    """Demonstrate encoding/decoding without PyMOO."""
    print("Demonstrating SearchSpaceRegistry encoding/decoding")
    print("=" * 60)
    print()

    # Load search space
    registry = SearchSpaceRegistry.from_yaml(
        "conf/search_space/speech_commands.yaml",
        validate=False,
        max_layers=12,
    )

    print(f"Loaded search space with {len(registry.layer_names)} layer types")
    print(f"Vector size: {registry.vector_size}")
    print()

    # Create a random chromosome
    print("Creating random chromosome...")
    chromosome = registry.create_random_chromosome()

    print(f"Generated {len(chromosome)} layers:")
    for gene in chromosome:
        print(f"  - {gene['layer']}: {gene.get('f_name', 'N/A')}")
    print()

    # Encode to vector
    print("Encoding to vector...")
    vector = registry.encode(chromosome)
    print(f"Vector shape: {vector.shape}")
    print(f"Vector min/max: {vector.min():.3f} / {vector.max():.3f}")
    print()

    # Decode back
    print("Decoding back to chromosome...")
    decoded = registry.decode(vector, enforce_rules=True)

    print(f"Decoded {len(decoded)} layers:")
    for gene in decoded:
        print(f"  - {gene['layer']}: {gene.get('f_name', 'N/A')}")
    print()

    # Verify roundtrip
    layers_match = [orig["layer"] == dec["layer"] for orig, dec in zip(chromosome, decoded)]
    print(f"Layer types preserved: {all(layers_match)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demonstrate_encoding()
    else:
        if PYMOO_AVAILABLE:
            run_example()
        else:
            print("PyMOO not available. Running encoding demo instead.")
            print()
            demonstrate_encoding()
