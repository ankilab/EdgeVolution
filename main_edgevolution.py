import os
import tensorflow as tf
import multiprocessing
import numpy as np
import hydra
from omegaconf import DictConfig

from neural_architecture_search.genetic_algorithm import GeneticAlgorithm
from neural_architecture_search.src.surrogate_model import SurrogateModel
from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry
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

    # Create surrogate model if enabled
    surrogate = None
    if hasattr(cfg, 'surrogate') and cfg.surrogate.enabled.value:
        surrogate = SurrogateModel(
            model_type=cfg.surrogate.model_type.value,
            n_estimators=cfg.surrogate.n_estimators.value,
            min_samples_to_train=cfg.surrogate.min_samples_to_train.value,
            confidence_threshold=cfg.surrogate.confidence_threshold.value,
            exploration_ratio=cfg.surrogate.exploration_ratio.value,
            evaluation_mode=cfg.surrogate.evaluation_mode.value,
        )

    if continue_path is not None:
        continue_generation = cfg.continue_generation
        my_loader = Loader(continue_path, continue_generation)
        cfg = my_loader.get_cfg()
        my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)
        my_ga = GeneticAlgorithm(cfg, my_saver, my_loader,
                                 surrogate=surrogate, search_space_registry=registry)
        gen_start = int(continue_generation)

        # Load surrogate state if it exists
        if surrogate is not None:
            surrogate_path = os.path.join(continue_path, 'surrogate')
            if os.path.exists(os.path.join(surrogate_path, 'metadata.json')):
                surrogate = SurrogateModel.load(surrogate_path)
                my_ga.surrogate = surrogate
    else:
        my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)
        my_ga = GeneticAlgorithm(cfg, my_saver,
                                 surrogate=surrogate, search_space_registry=registry)
        my_ga.init_first_generation()
        gen_start = 1

    # save params
    my_saver.save_params(cfg)

    _run_evolution(cfg, my_ga, gen_start)


def _run_evolution(cfg, my_ga, gen_start):
    use_mcu = cfg.hyperparameters.optimize_for_MCU.value

    if use_mcu:
        # update the tensor arena size in the main.cpp file
        limit = _get_tensor_arena_size_limit(cfg)
        update_tensor_arena_size("tflite/edgevolution_tflite/src/main_functions.cpp", limit)

    for i_generation in range(gen_start, cfg.hyperparameters.num_generations.value + 1):
        my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        my_ga.evaluate_memory_footprint()

        # Surrogate pre-screening: skip training for low-predicted-accuracy individuals
        my_ga.surrogate_prescreen()

        if use_mcu:
            # Evaluate candidate models on MCU (i.e. flash them to MCU and measure objectives)
            # this will start a process that is constantly running and evaluating an individual after training is finished
            process = multiprocessing.Process(target=my_ga.evaluate_energy_consumption_and_inference_speed)
            process.start()

        # train all neural networks
        my_ga.train_neural_networks()

        if use_mcu:
            # wait for the process to finish
            process.join()

        # Collect surrogate training data and merge skipped individuals back
        my_ga.collect_surrogate_data()

        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != cfg.hyperparameters.num_generations.value:
            my_ga.crossover()
            my_ga.mutation()


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
