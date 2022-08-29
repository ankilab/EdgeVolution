from genetic_algorithm import GeneticAlgorithm
from utils.saver import Saver

#################################
# define ENAS hyper-parameters
#################################
NB_GENERATIONS = 2
POPULATION_SIZE = 4
MUTATION_RATE = 100 #5  # in percent

MAX_NB_FEATURE_LAYERS = 8  # max number layers before GAP, GMP or Flatten layer
MAX_NB_CLASSIFICATION_LAYERS = 3  # max number layers after GAP, GMP or Flatten layer

PATH_GENE_POOL = "gene_pool.txt"
PATH_RULE_SET = "rule_set.txt"

#################################
# define DNN training hyper-parameters
#################################
INPUT_SHAPE = (32, 32, 1)
NB_CLASSES = 4
NB_EPOCHS = 15
BATCH_SIZE = 256

#################################
# define constraints
#################################
MAX_MEMORY_FOOTPRINT = 900000  # in Bytes (900000 bytes --> 0.9 MB)
MIN_INFERENCE_SPEED = 5  # in fps
MAX_POWER_CONSUMPTION = 1.4  # in mJ --> find out a reasonable number

#################################
# objectives weighting factors
#################################
acc_weight = 0.4  # accuracy
mem_weight = 0.2  # memory footprint
inf_weight = 0.2  # inference speed
enc_weight = 0.2  # energy consumption
# e = 1  # fine-tuning ability (was war nochmal genau gemeint damit?)

params = {'generations': NB_GENERATIONS,
          'population_size': POPULATION_SIZE,
          'mutation_rate': MUTATION_RATE,
          'max_nb_feature_layers': MAX_NB_FEATURE_LAYERS,
          'max_nb_classification_layers': MAX_NB_CLASSIFICATION_LAYERS,
          'path_gene_pool': PATH_GENE_POOL,
          'path_rule_set': PATH_RULE_SET,
          'input_shape': INPUT_SHAPE,
          'nb_classes': NB_CLASSES,
          'nb_epochs': NB_EPOCHS,
          'batch_size': BATCH_SIZE,
          'max_memory_footprint': MAX_MEMORY_FOOTPRINT,
          'max_inference_speed': MIN_INFERENCE_SPEED,
          'max_power_consumption': MAX_POWER_CONSUMPTION,
          'acc_weight': acc_weight,
          'mem_weight': mem_weight,
          'inf_weight': inf_weight,
          'enc_weight': enc_weight}


def main():
    my_saver = Saver()
    my_ga = GeneticAlgorithm(params, my_saver)

    # save params
    my_saver.save_params(params)

    # prepare dataset here
    # TODO: dataset preparation

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

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != NB_GENERATIONS:
            my_ga.selection()
            my_ga.crossover()
            my_ga.mutation()

        
if __name__ == "__main__":
    main()
