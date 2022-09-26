from genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver

#################################
# define which dataset and experiment to use (--> adjust it in train.py and add more datasets there)
#################################
SPEECH_COMMANDS_EXPERIMENTS = ["sc_ex_1", "sc_ex_2", "sc_ex_3", "sc_ex_4"]

# select what experiment to run --> make sure the data loader is defined in train.py
DATASET = SPEECH_COMMANDS_EXPERIMENTS[0]

#################################
# define ENAS hyper-parameters
#################################
NB_GENERATIONS = 10
POPULATION_SIZE = 100
NB_BEST_MODELS_CROSSOVER = 20  # specifies the number of models that will be used for crossover
MUTATION_RATE = 10  # in percent

MAX_NB_FEATURE_LAYERS = 8  # max number layers before GAP, GMP or Flatten layer in the first generation
MAX_NB_CLASSIFICATION_LAYERS = 3  # max number layers after GAP, GMP or Flatten layer in the first generation

PATH_GENE_POOL = "gene_pool.txt"
PATH_RULE_SET = "rule_set.txt"

#################################
# define DNN training hyper-parameters
#################################
INPUT_SHAPE = (16000, 1)
NB_CLASSES = 2
NB_EPOCHS = 10
BATCH_SIZE = 64
NB_MODELS_TRAINED_PARALLEL = 10

#################################
# define constraints
#################################
MAX_MEMORY_FOOTPRINT = 900000  # in Bytes (900000 bytes --> 0.9 MB)
MIN_INFERENCE_SPEED = 200  # in ms
MAX_POWER_CONSUMPTION = 1.4  # in mJ --> TODO: find out a reasonable number

params = {'dataset': DATASET,
          'generations': NB_GENERATIONS,
          'population_size': POPULATION_SIZE,
          'nb_best_models_crossover': NB_BEST_MODELS_CROSSOVER,
          'mutation_rate': MUTATION_RATE,
          'max_nb_feature_layers': MAX_NB_FEATURE_LAYERS,
          'max_nb_classification_layers': MAX_NB_CLASSIFICATION_LAYERS,
          'path_gene_pool': PATH_GENE_POOL,
          'path_rule_set': PATH_RULE_SET,
          'input_shape': INPUT_SHAPE,
          'nb_classes': NB_CLASSES,
          'nb_epochs': NB_EPOCHS,
          'batch_size': BATCH_SIZE,
          'nb_models_trained_parallel': NB_MODELS_TRAINED_PARALLEL,
          'max_memory_footprint': MAX_MEMORY_FOOTPRINT,
          'max_inference_speed': MIN_INFERENCE_SPEED,
          'max_power_consumption': MAX_POWER_CONSUMPTION}


def main():
    my_saver = Saver()
    my_ga = GeneticAlgorithm(params, my_saver)

    # save params
    my_saver.save_params(params)

    prepare_dataset("speech_commands_1")

    # random init the population of the first generation
    my_ga.init_first_generation()

    for i_generation in range(1, NB_GENERATIONS+1):
        my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        my_ga.evaluate_memory_footprint()

        # train all neural networks that actually fit into MCU flash memory
        my_ga.train_neural_networks()

        # Evaluate candidate models on MCU (i.e. flash them to MCU and measure objectives)
        #my_ga.evaluate_energy_consumption_and_inference_speed()

        # determine the fitness for each model and select the best ones
        my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != NB_GENERATIONS:
            my_ga.crossover()
            my_ga.mutation()

        
if __name__ == "__main__":
    main()
