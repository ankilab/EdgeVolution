import tensorflow as tf
import multiprocessing

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver

#################################
# define which dataset and experiment to use (--> adjust it in train.py and add more datasets there)
#################################
DATASETS = ["speech_commands"]
#EXPERIMENT = "fix_max_filesize"
EXPERIMENT = "FIRST_COMPLETE_RUN"

# select what dataset to use --> make sure the data loader is defined in datasets/get_datasets.py
DATASET = DATASETS[0]
SAMPLE_RATE = 16_000
INPUT_SHAPE = (6_000, 1)
NB_CLASSES = 12

#################################
# define ENAS hyper-parameters
#################################
NB_GENERATIONS = 40
POPULATION_SIZE = 100
NB_BEST_MODELS_CROSSOVER = 20  # specifies the number of models that will be used for crossover
MUTATION_RATE = 10  # in percent

MAX_NB_FEATURE_LAYERS = 30  # max number layers before GAP, GMP or Flatten layer in the first generation
MAX_NB_CLASSIFICATION_LAYERS = 6  # max number layers after GAP, GMP or Flatten layer in the first generation

PATH_GENE_POOL = "gene_pool.txt"
PATH_RULE_SET = "rule_set.txt"

# number of samples that will be averaged when measuring power consumption
POWER_MEASUREMENT_NB_SAMPLES_AVERAGE = 2_000

# threshold in mA that is used after the average filter was applied (i.e., value above 'threshold' is the start,
# where inference started, the next value below 'threshold' is the end of inference)
POWER_MEASUREMENT_THRESHOLD = 2200

#################################
# define DNN training hyper-parameters
#################################
NB_EPOCHS = 10
MIN_FREE_SPACE_GPU = 5_000_000_000  # 5 GB

#################################
# define constraints
#################################
MAX_MEMORY_FOOTPRINT = 900_000  # in Bytes (900000 bytes --> 0.9 MB)
MIN_INFERENCE_TIME = 300  # in ms
MAX_ENERGY_CONSUMPTION = 3  # in mJ

params = {'dataset': DATASET,
          'sample_rate': SAMPLE_RATE,
          'generations': NB_GENERATIONS,
          'population_size': POPULATION_SIZE,
          'nb_best_models_crossover': NB_BEST_MODELS_CROSSOVER,
          'mutation_rate': MUTATION_RATE,
          'max_nb_feature_layers': MAX_NB_FEATURE_LAYERS,
          'max_nb_classification_layers': MAX_NB_CLASSIFICATION_LAYERS,
          'power_measurement_nb_samples_average': POWER_MEASUREMENT_NB_SAMPLES_AVERAGE,
          'power_measurement_threshold': POWER_MEASUREMENT_THRESHOLD,
          'path_gene_pool': PATH_GENE_POOL,
          'path_rule_set': PATH_RULE_SET,
          'input_shape': INPUT_SHAPE,
          'nb_classes': NB_CLASSES,
          'nb_epochs': NB_EPOCHS,
          'min_free_space_gpu': MIN_FREE_SPACE_GPU,
          'max_memory_footprint': MAX_MEMORY_FOOTPRINT,
          'max_inference_time': MIN_INFERENCE_TIME,
          'max_energy_consumption': MAX_ENERGY_CONSUMPTION}


def main():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_saver = Saver(EXPERIMENT)
    my_ga = GeneticAlgorithm(params, my_saver)

    # save params
    my_saver.save_params(params)

    # random init the population of the first generation
    my_ga.init_first_generation()

    for i_generation in range(1, NB_GENERATIONS+1):
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

        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != NB_GENERATIONS:
            my_ga.crossover()
            my_ga.mutation()


if __name__ == "__main__":
    main()
