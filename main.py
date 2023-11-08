import tensorflow as tf
import multiprocessing
import argparse

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver
from utils.loader import Loader

#################################
# define which dataset and experiment to use (--> adjust it in train.py and add more datasets there)
#################################
DATASETS = ["speech_commands", "motion_sense_accelerometer"]
EXPERIMENT = "sc_2d_4classes_selection50_mutation20"
# EXPERIMENT = "test"

# select what dataset to use --> make sure the data loader is defined in datasets/get_datasets.py
DATASET = DATASETS[0]
SAMPLE_RATE = 16_000
INPUT_SHAPE = (32,32,3)
NB_CLASSES = 10  # Speech commands is a 12 classes problem
CLASSES_FILTER = [0, 2, 6, 8]  # containing all classes that should be used for optimization
# CLASSES_FILTER = []  # empty --> all classes are used

if len(CLASSES_FILTER) != 0:
    NB_CLASSES = len(CLASSES_FILTER)

#################################
# define EvoNAS hyper-parameters
#################################
NB_GENERATIONS = 2
POPULATION_SIZE = 2
NB_BEST_MODELS_CROSSOVER = 5  # specifies the number of models that will be used for crossover
MUTATION_RATE = 20  # in percent

MAX_NB_FEATURE_LAYERS = 10  # max number layers before GAP, GMP or Flatten layer in the first generation
MAX_NB_CLASSIFICATION_LAYERS = 6  # max number layers after GAP, GMP or Flatten layer in the first generation

PATH_GENE_POOL = "gene_pool.txt"
PATH_RULE_SET = "rule_set.txt"


######################################
# define DNN training hyper-parameters
######################################
NB_EPOCHS = 1
MIN_FREE_SPACE_GPU = 6_000_000_000  # 6 GB


params = {'dataset': DATASET,
          'sample_rate': SAMPLE_RATE,
          'generations': NB_GENERATIONS,
          'population_size': POPULATION_SIZE,
          'nb_best_models_crossover': NB_BEST_MODELS_CROSSOVER,
          'mutation_rate': MUTATION_RATE,
          'max_nb_feature_layers': MAX_NB_FEATURE_LAYERS,
          'max_nb_classification_layers': MAX_NB_CLASSIFICATION_LAYERS,
          'path_gene_pool': PATH_GENE_POOL,
          'path_rule_set': PATH_RULE_SET,
          'input_shape': INPUT_SHAPE,
          'classes_filter': CLASSES_FILTER,
          'nb_classes': NB_CLASSES,
          'nb_epochs': NB_EPOCHS,
          'min_free_space_gpu': MIN_FREE_SPACE_GPU,}


def main(continue_from=None):
    my_saver = Saver(EXPERIMENT)

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

    for i_generation in range(gen_start, NB_GENERATIONS+1):
        my_ga.prepare_generation(i_generation)
        print("now training the models")
        # train all neural networks that actually fit into MCU flash memory
        my_ga.train_neural_networks()

        # determine the fitness for each model and select the best ones
        #my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != NB_GENERATIONS:
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
    main(continue_from)
