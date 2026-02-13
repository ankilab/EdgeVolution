import tensorflow as tf
import multiprocessing
import numpy as np
import hydra
from omegaconf import DictConfig

from neural_architecture_search.genetic_algorithm import GeneticAlgorithm
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
    if continue_path is not None:
        continue_generation = cfg.continue_generation
        my_loader = Loader(continue_path, continue_generation)
        cfg = my_loader.get_cfg()
        my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)
        my_ga = GeneticAlgorithm(cfg, my_saver, my_loader)
        gen_start = int(continue_generation)
    else:
        my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)
        my_ga = GeneticAlgorithm(cfg, my_saver)
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
