import os.path
from subprocess import Popen
from coolname import generate_slug
import json

from src.genepool import GenePool
from src.translation import translate
from utils.helper import grouper
from utils.saver import Saver
from src.crossover import crossover

from utils.convert_to_tflite import convert_to_tflite
from tflite.flash_tflite_model import flash_tflite_model


class GeneticAlgorithm:
    def __init__(self, params: dict, saver: Saver):
        self.params = params
        self.my_saver = saver
        self.my_gene_pool = GenePool(params)

        # variables that change after each generation
        self.individuals_names = None  # randomly created names for all individuals within one generation
        self.preselected_individuals = None  # preselected individuals (names) after accuracy/memory footprint determination
        self.population_genotype = None  # dicts containing all individuals with its properties
        self.population_phenotype = None  # generated TF models
        self.population_phenotype_tflite = None  # generated TFLite models
        self.generation_counter = None  # information in which generation we are currently
        self.population_next_generation = None  # contains all individuals for next generation (determined through selection, crossover and mutation)
        self.best_models_current_generation = None  # contains the best models of the current generation

    def init_first_generation(self):
        self.population_genotype = []
        for _ in range(self.params['population_size']):
            random_chromosome = self.my_gene_pool.get_random_chromosome()
            self.population_genotype.append(random_chromosome)

    def evaluate_memory_footprint(self):
        """ Load memory footprint after converting the model to TFLite.
        Determine models (dependent on the thresholds specified in main.py) that will be further evaluated. """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        for individual in self.individuals_names:
            # read TFLite model memory footprint
            tflite_model_path = path + individual + '/models/model_tflite_untrained.tflite'
            memory_footprint = os.path.getsize(tflite_model_path)
            d = {'memory_footprint': memory_footprint}
            with open(path + individual + '/results.json', 'w') as f:
                json.dump(d, f, indent=2)

            # delete the untrained TFLite model
            os.remove(tflite_model_path)

            # take only the models into account that are below a certain threshold
            if memory_footprint <= self.params['max_memory_footprint']:
                self.preselected_individuals.append(individual)

    def train_neural_networks(self):
        # train all neural networks
        # --> start several training processes here
        for individuals in grouper(5, self.preselected_individuals):
            command = 'python genetic_algorithm/src/train.py' + \
                      f' {self.my_saver.results_dir} Generation_{self.generation_counter}'
            procs = [Popen(command + ' ' + name, shell=True) for name in individuals if name is not None]
            for p in procs:
                p.wait()

    def evaluate_energy_consumption_and_inference_speed(self):
        """ Evaluate all preselected models on the MCU. """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        for individual in self.preselected_individuals:
            # determine energy consumption and inference speed
            flash_tflite_model(path + individual + '/models/model_trained.h5')

    def selection(self):
        # calculate fitness of all preselected models --> order them by their achieved fitness
        # for individual in self.preselected_individuals:
        #    with open() as f:
        #        d = json.loads(f.read())
        #        fitness = calculate_fitness()
        #
        #        # save fitness in results.json
        #        json.dump(d, f)
        #        self.best_models.append()
        self.best_models_current_generation = self.preselected_individuals

    def crossover(self):
        self.population_next_generation = crossover(
            path=f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/',
            fittest_chromosomes=self.best_models_current_generation,
            population_size=self.params['population_size'])

    def mutation(self):
        self.population_genotype = []
        for chromosome in self.population_next_generation:
            mutated_chromosome = self.my_gene_pool.mutate_chromosome(chromosome)
            self.population_genotype.append(mutated_chromosome)

    def prepare_generation(self, current_generation: int):
        self.individuals_names = []
        self.population_phenotype = []
        self.population_phenotype_tflite = []
        self.preselected_individuals = []
        self.best_models_current_generation = []
        self.population_next_generation = []

        self._generate_individuals_names()

        self.generation_counter = current_generation

        # translate all chromosomes: genotype (chromosome) --> phenotype (tf.keras.Model)
        for chromosome in self.population_genotype:
            model = translate(chromosome, self.params['input_shape'], self.params['nb_classes'])
            tflite_model = convert_to_tflite(model)
            self.population_phenotype.append(model)
            self.population_phenotype_tflite.append(tflite_model)

        # save untrained networks such that it can be loaded in a new process
        self.my_saver.save_chromosomes(self.population_genotype, self.population_phenotype,
                                       self.population_phenotype_tflite, self.individuals_names,
                                       self.generation_counter)

    def _generate_individuals_names(self):
        self.individuals_names = []
        for _ in range(self.params['population_size']):
            # Assigning a random name to an individual to make it easier to track results
            random_name = generate_slug(2).replace("-", "_")
            while random_name in self.individuals_names:  # --> make sure that the random name does not already exist
                random_name = generate_slug(2).replace("-", "_")
            self.individuals_names.append(random_name)
