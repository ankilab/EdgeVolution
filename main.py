import tensorflow as tf
import multiprocessing
import argparse
import numpy as np
import hydra
from omegaconf import DictConfig

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver
from utils.loader import Loader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Limit TensorFlow GPU memory usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)

    my_ga = GeneticAlgorithm(cfg, my_saver)

    # random init the population of the first generation
    my_ga.init_first_generation()
    gen_start = 1

    # save params
    my_saver.save_params(cfg)

    for i_generation in range(gen_start, cfg.hyperparameters.num_generations.value + 1):
        my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        my_ga.evaluate_memory_footprint()

        if cfg.hyperparameters.optimize_for_MCU.value:
            # Evaluate candidate models on MCU (i.e. flash them to MCU and measure objectives)
            # this will start a process that is constantly running and evaluating an individual after training is finished
            process = multiprocessing.Process(target=my_ga.evaluate_energy_consumption_and_inference_speed)
            process.start()

        # train all neural networks 
        my_ga.train_neural_networks()

        if cfg.hyperparameters.optimize_for_MCU.value:
            # wait for the process to finish
            process.join()

        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != cfg.hyperparameters.num_generations.value:
            my_ga.crossover()
            my_ga.mutation()


if __name__ == "__main__":
    np.random.seed(42)
    main()
