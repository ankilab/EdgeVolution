import tensorflow as tf
import multiprocessing
import argparse
import yaml
import numpy as np

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver
from utils.loader import Loader

with open("config.yaml", 'r') as yaml_file:
    params = yaml.safe_load(yaml_file)

if len(params["classes_filter"]) != 0:
    params["nb_classes"] = len(params["classes_filter"])


def main(continue_from=None):
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_saver = Saver(params["experiment_description"])

    if continue_from['continue_from_ga_run'] is None:
        my_ga = GeneticAlgorithm(params, my_saver)
        # random init the population of the first generation
        my_ga.init_first_generation()
        gen_start = 1

        # save params
        my_saver.save_params(params)
    else:
        my_loader = Loader(continue_from)
        params_ = my_loader.get_params()

        gen_start = my_loader.get_gen_start()

        my_ga = GeneticAlgorithm(params_, my_saver, my_loader)

        # save params
        my_saver.save_params(params_)

    for i_generation in range(gen_start, params["nb_generations"] + 1):
        my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        my_ga.evaluate_memory_footprint()

        # Evaluate candidate models on MCU (i.e. flash them to MCU and measure objectives)
        # this will start a process that is constantly running and evaluating an individual after training is finished
        process = multiprocessing.Process(target=my_ga.evaluate_energy_consumption_and_inference_speed)
        process.start()

        # train all neural networks that actually fit into MCU flash memory
        my_ga.train_neural_networks()

        # wait for the process to finish
        process.join(timeout=5)
        process.terminate()

        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != params["nb_generations"]:
            my_ga.crossover()
            my_ga.mutation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Evolutionary Neural Architecture Search',
        description='This package runs an evolutionary neural architecture search to find constrained DNN architectures.'
    )

    parser.add_argument('continue_from_ga_run', nargs='?', default=None)
    parser.add_argument('continue_from_generation', nargs='?', default=None)
    args = parser.parse_args()

    continue_from = {'continue_from_ga_run': args.continue_from_ga_run,
                     'continue_from_generation': args.continue_from_generation}

    np.random.seed(42)

    main(continue_from)
