"""
EdgeVolution main entry point with pluggable search strategies.

Uses an ask/tell interface to decouple search strategies from the
evaluation pipeline:

    for each generation:
        candidates = strategy.ask(n)
        results = pipeline.evaluate(candidates)
        strategy.tell(results)

Usage:
    python main.py +hyperparameters=speech_commands +search_space=speech_commands +boards=none

Select a search strategy:
    python main.py ... search_strategy=genetic_algorithm   (default)
    python main.py ... search_strategy=random
    python main.py ... search_strategy=regularized_evolution
    python main.py ... search_strategy=pymoo
    python main.py ... search_strategy=ax
"""

import os
import json
import numpy as np
import tensorflow as tf
import hydra
from omegaconf import DictConfig

from neural_architecture_search.strategies import create_strategy
from neural_architecture_search.evaluation_pipeline import EvaluationPipeline
from neural_architecture_search.src.surrogate_model import SurrogateModel
from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry
from neural_architecture_search.src.hardware_lut import HardwareLUT
from utils.saver import Saver
from utils.loader import Loader
from tools.update_tensor_arena_size import update_tensor_arena_size


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Limit TensorFlow GPU memory usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    continue_path = cfg.continue_path

    # Create search space registry
    registry = None
    if hasattr(cfg, 'search_space'):
        registry = SearchSpaceRegistry.from_omegaconf(cfg.search_space, validate=False)

    # ------------------------------------------------------------------
    # Backward compatibility: accept old 'surrogate' key as 'surrogate_accuracy'
    # ------------------------------------------------------------------
    if hasattr(cfg, 'surrogate') and not hasattr(cfg, 'surrogate_accuracy'):
        cfg.surrogate_accuracy = cfg.surrogate

    # ------------------------------------------------------------------
    # Create accuracy surrogate
    # ------------------------------------------------------------------
    surrogate = _create_surrogate(cfg, 'surrogate_accuracy', target_name='accuracy')

    # ------------------------------------------------------------------
    # Create hardware surrogate
    # ------------------------------------------------------------------
    hw_surrogate = _create_surrogate(cfg, 'surrogate_hardware', target_name='energy')

    # ------------------------------------------------------------------
    # Load hardware LUT (if configured)
    # ------------------------------------------------------------------
    hw_lut = None
    if hasattr(cfg, 'hardware_lut') and cfg.hardware_lut.enabled.value:
        lut_path = cfg.hardware_lut.path.value
        if lut_path is not None:
            hw_lut = HardwareLUT.load(lut_path)
            print(f"Loaded hardware LUT from {lut_path} (mode={hw_lut.mode})")

    if continue_path is not None:
        continue_generation = int(cfg.continue_generation)
        my_loader = Loader(continue_path, continue_generation)
        cfg = my_loader.get_cfg()
        my_saver = Saver.from_existing(continue_path)

        # Load surrogate state if it exists
        if surrogate is not None:
            surrogate_path = os.path.join(continue_path, 'surrogate_accuracy')
            # Also check legacy 'surrogate' directory
            if not os.path.exists(os.path.join(surrogate_path, 'metadata.json')):
                surrogate_path = os.path.join(continue_path, 'surrogate')
            if os.path.exists(os.path.join(surrogate_path, 'metadata.json')):
                surrogate = SurrogateModel.load(surrogate_path)

        if hw_surrogate is not None:
            hw_surrogate_path = os.path.join(continue_path, 'surrogate_hardware')
            if os.path.exists(os.path.join(hw_surrogate_path, 'metadata.json')):
                hw_surrogate = SurrogateModel.load(hw_surrogate_path)

        # Create strategy and pipeline
        strategy = create_strategy(cfg, registry)
        pipeline = EvaluationPipeline(
            cfg, my_saver,
            surrogate=surrogate,
            hw_surrogate=hw_surrogate,
            search_space_registry=registry,
            hw_lut=hw_lut,
        )

        # Complete any missing MCU evaluations, then seed strategy
        use_mcu = cfg.hyperparameters.optimize_for_MCU.value
        if use_mcu:
            limit = _get_tensor_arena_size_limit(cfg)
            update_tensor_arena_size("tflite/edgevolution_tflite/src/main_functions.cpp", limit)
            _complete_mcu_evaluation(pipeline, cfg, continue_path, continue_generation)

        gen_start = _seed_strategy_from_disk(
            strategy, cfg, continue_path, continue_generation, my_saver,
        )
    else:
        my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)
        gen_start = 1

        # Create strategy and pipeline
        strategy = create_strategy(cfg, registry)
        pipeline = EvaluationPipeline(
            cfg, my_saver,
            surrogate=surrogate,
            hw_surrogate=hw_surrogate,
            search_space_registry=registry,
            hw_lut=hw_lut,
        )

        # Save params
        my_saver.save_params(cfg)

    _run_search(cfg, strategy, pipeline, my_saver, gen_start)


def _create_surrogate(cfg, config_key, target_name='accuracy'):
    """Create a SurrogateModel from the config section named *config_key*.

    Handles three cases:
    1. ``pretrained_path`` is set → load from previous run with config overrides.
    2. ``enabled`` is True → create a fresh SurrogateModel.
    3. Otherwise → return None.
    """
    if not hasattr(cfg, config_key):
        return None

    section = getattr(cfg, config_key)
    enabled = section.enabled.value
    pretrained_path = getattr(section, 'pretrained_path', None)
    if pretrained_path is not None:
        pretrained_path = pretrained_path.value

    if pretrained_path:
        return SurrogateModel.load_pretrained(
            pretrained_path,
            confidence_threshold=section.confidence_threshold.value,
            exploration_ratio=section.exploration_ratio.value,
            evaluation_mode=section.evaluation_mode.value,
            target_name=target_name,
        )
    elif enabled:
        return SurrogateModel(
            model_type=section.model_type.value,
            n_estimators=section.n_estimators.value,
            min_samples_to_train=section.min_samples_to_train.value,
            confidence_threshold=section.confidence_threshold.value,
            exploration_ratio=section.exploration_ratio.value,
            evaluation_mode=section.evaluation_mode.value,
            target_name=target_name,
        )
    return None


def _apply_population_size_decay(cfg, generation):
    """Resolve the population_size_decay schedule for the current generation."""
    decay = cfg.hyperparameters.population_size_decay.value
    cfg.hyperparameters.population_size.value = next(
        s[1] for s in decay[::-1] if generation >= s[0]
    )


def _complete_mcu_evaluation(pipeline, cfg, continue_path, continue_generation):
    """Re-run MCU evaluation for individuals missing energy data."""
    gen_dir = os.path.join(continue_path, f'Generation_{continue_generation}')
    if not os.path.isdir(gen_dir):
        return

    snr = cfg.boards.value[0].snr if len(cfg.boards.value) > 0 else None
    if snr is None:
        return

    incomplete = {}
    for name in sorted(os.listdir(gen_dir)):
        ind_dir = os.path.join(gen_dir, name)
        if not os.path.isdir(ind_dir):
            continue
        results_path = os.path.join(ind_dir, 'results.json')
        chromosome_path = os.path.join(ind_dir, 'chromosome.json')
        if not os.path.exists(results_path) or not os.path.exists(chromosome_path):
            continue

        with open(results_path) as f:
            results = json.load(f)

        energy_info = results.get('energy_information')
        has_energy = (isinstance(energy_info, dict)
                      and any(isinstance(v, (int, float)) for v in energy_info.values()))
        if not has_energy:
            with open(chromosome_path) as f:
                chromosome = json.load(f)
            incomplete[name] = {"genotype": chromosome}

    if not incomplete:
        print(f"All individuals in Generation_{continue_generation} have MCU results.")
        return

    print(f"Re-running MCU evaluation for {len(incomplete)}/{len(os.listdir(gen_dir))} "
          f"individuals in Generation_{continue_generation}...")
    pipeline._evaluate_mcu(incomplete, continue_generation)

    # Recalculate fitness for the re-evaluated individuals
    from neural_architecture_search.src.objective_function import calculate_fitness
    for name in incomplete:
        results_path = os.path.join(gen_dir, name, 'results.json')
        with open(results_path) as f:
            results = json.load(f)
        fitness, error = calculate_fitness(results, cfg)
        results['fitness'] = float(fitness)
        results['error'] = str(error)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"MCU evaluation completed for {len(incomplete)} individuals.")


def _seed_strategy_from_disk(strategy, cfg, continue_path, continue_generation, saver):
    """Reconstruct strategy state from saved results on disk.

    Loads individuals and their results from the continue_generation directory,
    builds EvaluationResult objects, and calls strategy.tell() to seed the
    strategy's internal state (e.g. fittest individuals for crossover).

    Returns the generation number to start the search loop from.
    """
    from neural_architecture_search.strategies.base import EvaluationResult
    from neural_architecture_search.src.objective_function import calculate_fitness

    gen_dir = os.path.join(continue_path, f'Generation_{continue_generation}')
    if not os.path.isdir(gen_dir):
        print(f"WARNING: Generation_{continue_generation} not found at {gen_dir}, "
              f"starting fresh from generation {continue_generation}")
        return continue_generation

    # Load all individuals from the generation
    results_list = []
    for name in sorted(os.listdir(gen_dir)):
        ind_dir = os.path.join(gen_dir, name)
        if not os.path.isdir(ind_dir):
            continue

        chromosome_path = os.path.join(ind_dir, 'chromosome.json')
        results_path = os.path.join(ind_dir, 'results.json')

        if not os.path.exists(chromosome_path) or not os.path.exists(results_path):
            continue

        with open(chromosome_path) as f:
            chromosome = json.load(f)
        with open(results_path) as f:
            results = json.load(f)

        fitness, error = calculate_fitness(results, cfg)

        energy_info = results.get('energy_information')
        energy = None
        if isinstance(energy_info, dict):
            vals = [v for v in energy_info.values() if isinstance(v, (int, float))]
            energy = vals[0] if vals else None

        inf_info = results.get('inference_information')
        inf_time = None
        if isinstance(inf_info, dict):
            vals = [v for v in inf_info.values() if isinstance(v, (int, float))]
            inf_time = vals[0] if vals else None

        results_list.append(EvaluationResult(
            name=name,
            chromosome=chromosome,
            val_acc=results.get('val_acc', 0.0),
            memory_footprint_tflite=results.get('memory_footprint_tflite', 0),
            fitness=fitness,
            energy_consumption=energy,
            inference_time=inf_time,
            rom_usage=results.get('rom_usage'),
            surrogate_predicted=results.get('surrogate_skipped', False),
            hw_surrogate_predicted=results.get('hw_surrogate_skipped', False),
            error=error,
        ))

    if results_list:
        # Apply decay schedules for the continue_generation so tell() uses
        # the correct num_best_models_crossover value.
        _apply_population_size_decay(cfg, continue_generation)
        strategy.on_generation_start(continue_generation)

        # For PyMOO: encode chromosomes into _pending_vectors before tell()
        if hasattr(strategy, '_pending_vectors') and hasattr(strategy, 'registry'):
            import numpy as np
            vectors = np.array([
                strategy.registry.encode(r.chromosome) for r in results_list
            ])
            strategy._pending_vectors = vectors

        strategy.tell(results_list)

        # Record best individual if not already in CSV
        best = strategy.get_best(1)
        if best:
            saver.save_best_individual(continue_generation, best[0].name, best[0].fitness)

        strategy.on_generation_end(continue_generation)
        print(f"Seeded strategy with {len(results_list)} results from "
              f"Generation_{continue_generation} (best fitness: {best[0].fitness:.4f})")
        return continue_generation + 1

    print(f"WARNING: No results found in Generation_{continue_generation}, "
          f"starting from generation {continue_generation}")
    return continue_generation


def _run_search(cfg, strategy, pipeline, saver, gen_start):
    """Main ask/tell search loop."""
    use_mcu = cfg.hyperparameters.optimize_for_MCU.value

    if use_mcu:
        limit = _get_tensor_arena_size_limit(cfg)
        update_tensor_arena_size("tflite/edgevolution_tflite/src/main_functions.cpp", limit)

    for generation in range(gen_start, cfg.hyperparameters.num_generations.value + 1):
        _apply_population_size_decay(cfg, generation)
        strategy.on_generation_start(generation)

        n = cfg.hyperparameters.population_size.value
        candidates = strategy.ask(n)

        results = pipeline.evaluate(candidates, generation, use_mcu=use_mcu)

        strategy.tell(results)

        best = strategy.get_best(1)
        if best:
            saver.save_best_individual(generation, best[0].name, best[0].fitness)

        strategy.on_generation_end(generation)


def _get_tensor_arena_size_limit(cfg):
    board_available_tensor_arena_size = cfg.boards.value[0].max_available_tensor_arena_size
    limit_tensor_arena_size = cfg.hyperparameters.limit_tensor_arena_size.value
    if limit_tensor_arena_size is None:
        limit_tensor_arena_size = board_available_tensor_arena_size
    else:
        limit_tensor_arena_size = min(limit_tensor_arena_size, board_available_tensor_arena_size)
    return limit_tensor_arena_size


if __name__ == "__main__":
    np.random.seed(42)
    main()
