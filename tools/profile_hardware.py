"""
Hardware profiling tool for building a Hardware Lookup Table (LUT).

Evaluates N random architectures on real MCU hardware, then builds a
HardwareLUT that can replace real MCU evaluation in future NAS runs.

Usage::

    python tools/profile_hardware.py \\
        +hyperparameters=speech_commands +search_space=speech_commands \\
        +boards=nrf52840dk \\
        hardware_profile.n_samples=200 \\
        hardware_profile.output=hardware_luts/nrf52840dk/ \\
        hardware_profile.mode=full
"""

import os
import sys
import json

import numpy as np
import tensorflow as tf
import hydra
from omegaconf import DictConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_architecture_search.strategies import create_strategy
from neural_architecture_search.evaluation_pipeline import EvaluationPipeline
from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry
from neural_architecture_search.src.hardware_lut import HardwareLUT
from utils.saver import Saver
from tools.update_tensor_arena_size import update_tensor_arena_size


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Limit TensorFlow GPU memory usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    n_samples = cfg.hardware_profile.n_samples.value
    output_path = cfg.hardware_profile.output.value
    mode = cfg.hardware_profile.mode.value

    if output_path is None:
        print("ERROR: hardware_profile.output must be set.")
        sys.exit(1)

    print(f"Hardware Profiling: {n_samples} random architectures, mode={mode}")
    print(f"Output: {output_path}")

    # Create search space registry
    registry = None
    if hasattr(cfg, 'search_space'):
        registry = SearchSpaceRegistry.from_omegaconf(cfg.search_space, validate=False)

    # Override population size for profiling
    cfg.hyperparameters.population_size.value = n_samples

    # Use random search strategy for profiling
    cfg.search_strategy = {"name": "random"}

    # Create saver and strategy
    saver = Saver(
        cfg.hyperparameters.results_path.value,
        f"profile_{cfg.hyperparameters.dataset_name.value}",
    )
    saver.save_params(cfg)

    strategy = create_strategy(cfg, registry)
    pipeline = EvaluationPipeline(
        cfg, saver, search_space_registry=registry,
    )

    # Set up MCU evaluation
    use_mcu = cfg.hyperparameters.optimize_for_MCU.value
    if use_mcu:
        board_available = cfg.boards.value[0].max_available_tensor_arena_size
        limit = cfg.hyperparameters.limit_tensor_arena_size.value
        if limit is None:
            limit = board_available
        else:
            limit = min(limit, board_available)
        update_tensor_arena_size(
            "tflite/edgevolution_tflite/src/main_functions.cpp", limit,
        )

    # Generate random architectures
    generation = 1
    strategy.on_generation_start(generation)
    candidates = strategy.ask(n_samples)

    # Convert to internal dict format
    individuals = {c.name: {"genotype": c.chromosome} for c in candidates}

    # Translate chromosomes → Keras models → TFLite (no training needed)
    pipeline._prepare_generation(individuals, generation)
    pipeline._evaluate_memory_footprint(individuals, generation)

    # MCU evaluation only — skip training (LUT only needs hardware metrics
    # measured on the untrained TFLite model)
    if use_mcu:
        print(f"Evaluating {len(individuals)} architectures on MCU (no training)...")
        pipeline._evaluate_mcu(individuals, generation)

    # Build results
    results = pipeline._build_results(individuals, generation)
    strategy.tell(results)
    strategy.on_generation_end(generation)

    best = strategy.get_best(1)
    if best:
        saver.save_best_individual(generation, best[0].name, best[0].fitness)

    print(f"\nProfiling complete. {len(results)} architectures evaluated.")
    print(f"Results saved to: {saver.results_dir}")

    # Build the LUT
    board_snr = cfg.boards.value[0].snr if len(cfg.boards.value) > 0 else None
    if board_snr is None:
        print("WARNING: No board SNR found. Cannot build LUT.")
        return

    print(f"\nBuilding Hardware LUT (mode={mode})...")
    lut = HardwareLUT.build_from_results(
        str(saver.results_dir), board_snr, registry, mode=mode,
    )

    # Cross-validation
    print("\nCross-validation results:")
    cv_results = lut.cross_validate(
        str(saver.results_dir), board_snr, registry, n_folds=5,
    )
    for metric, scores in cv_results.items():
        print(f"  {metric}: R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}")

    # Save
    lut.save(output_path)
    print(f"\nHardware LUT saved to: {output_path}")
    print(f"\nTo use in a NAS run:")
    print(f"  python main.py ... hardware_lut.enabled.value=true "
          f"hardware_lut.path.value={output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
