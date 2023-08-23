import os.path
import signal
from subprocess import Popen

import pandas as pd
from coolname import generate_slug
import json
import numpy as np
import time
import subprocess
import nvidia_smi
from multiprocessing import Pool

from .src.genepool import GenePool
from .src.translation import translate
from utils.saver import Saver
from utils.loader import Loader
from .src.fitness import calculate_fitness

from .utils.convert_to_tflite import convert_to_tflite
from .utils.substitute_tflite_layer import substitute_tflite_layer
from tools.measure_power_consumption import init_ppk2, stop_measuring
from multiprocessing import get_context


class GeneticAlgorithm:
    def __init__(self, params: dict, saver: Saver, loader: Loader=None):
        self.params = params
        self.my_saver = saver
        self.my_gene_pool = GenePool(params)

        # define all variables that change after each generation
        if loader is None:
            self.population_genotype = None  # dicts containing all individuals with its properties
        else:
            # load all genotypes --> each of them will get a new name but that's the easiest solution I found for now
            self.population_genotype = loader.load_population_genotype()

        self.individuals_names = None  # randomly created names for all individuals within one generation
        self.preselected_individuals = None  # preselected individuals (names) after accuracy/memory footprint determination
        self.parents_names = None  # list containing the chromosome names of the parents that yielded to a new chromosome
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
            # read h5 model memory footprint
            model_path = path + individual + '/models/model_untrained.h5'
            memory_footprint_h5 = os.path.getsize(model_path)

            # read TFLite model memory footprint
            tflite_model_path = path + individual + '/models/model_tflite_untrained.tflite'
            memory_footprint_tflite = os.path.getsize(tflite_model_path)

            # read C-Array memory footprint
            c_array_path = path + individual + '/models/model_c_array_untrained.cc'
            memory_footprint_c_array = os.path.getsize(c_array_path)

            d = {'memory_footprint_h5': memory_footprint_h5,
                 'memory_footprint_tflite': memory_footprint_tflite,
                 'memory_footprint_c_array': memory_footprint_c_array}

            # delete the untrained TFLite model and C array
            os.remove(c_array_path)

            # take only the models into account that are below a certain threshold
            if memory_footprint_tflite <= self.params['max_memory_footprint']:
                self.preselected_individuals.append(individual)
            else:
                # set fitness directly to zero since it is not relevant anymore
                d["fitness"] = 0

            with open(path + individual + '/results.json', 'w') as f:
                json.dump(d, f, indent=2)

        if len(self.preselected_individuals) == 0:
            raise Exception("All models are too big in terms of file size. Therefore none of the generated models will"
                            " be further evaluated. Think about adjusting your GA parameters.")

    def train_neural_networks(self):
        # train all neural networks
        # --> start several training processes here
        min_free_space = self.params['min_free_space_gpu']
        procs = []
        idx = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        while idx < len(self.preselected_individuals):
            procs = [p for p in procs if p.poll() is None]
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            if info.free > min_free_space:
                command = 'python genetic_algorithm/src/train.py ' + \
                           f'--results_dir {self.my_saver.results_dir} ' + \
                           f'--gen_dir Generation_{self.generation_counter} ' + \
                           f'--individual_dir {self.preselected_individuals[idx]} ' + \
                           f'--nb_epochs {self.params["nb_epochs"]} ' + \
                           f'--dataset {self.params["dataset"]} ' + \
                           f'--classes_filter ' + ' '.join(str(i) for i in self.params["classes_filter"])
                procs.append(Popen(command, shell=True))
                idx += 1

            time.sleep(15)

        nvidia_smi.nvmlShutdown()
        # make sure to wait until all processes are finished
        for p in procs:
            p.wait()

    def get_inference_information_from_results(self, board_snr, results):
        """ 
        get_inference_information_from_results reads the inference information from results and provides some checking. 

        :param board_snr: string containing information the board snr
        :param results: dict that contains the inference_information of all the measured boards
        
        :raises: ValueError if result does not contain key 'inference_information' or inference_information does not contain the provided board_snr
        :return: Inference time if exists
        """ 
        
        # check for inference_information in results
        if "inference_information" not in results:
            raise ValueError("key 'inference_information' does not exist")
        
        # check for board_snr in inference_information
        if board_snr not in results["inference_information"]:
            raise ValueError("board_snr does not exist in inference information")
        
        # raises valueError if not possible to be converted to float
        value = float(results["inference_information"][board_snr])

        # return the inference time of the provided board_snr
        return value



    def set_result_value_for_board(self, board_snr, category, value, results):
        """ 
        update_or_set_result_value checks if the board information is already available and appends it to the dict
        structure of "<type>_information" is expected to be this:
        "<type>_information" : {
                    "board_1" : [board_1_information],
                    "board_2: : [board_2_information],
                    ....
        }

        :param board_snr: string containing information the board snr
        :param category: string that specifies the type of results key. e.g. "energy_information" or "mean_power_information"
        :param value: the value that will be added to the information for the category
        :param result: the current data that has already been saved

        :raises: RuntimeError if the category with board_snr is already set
        :return: dictionary with updated information
        """ 
        
        # board snr should be unique id that serves as key for the information
        id = board_snr

        # get previous category information if exists
        information = {}
        if category in results:

            #get old information of category
            information = results[category]

            # inference information of specified board should not already be contained beforehand
            if id in information:
                raise RuntimeError(f"result.json already contains {category} of the specified board {board_snr}")
            

        # append new information for board with id
        information[id] = value

        # return the updated information
        return information 


    def calculate_energy_consumption(self, board_snr:str, data_dir:str):
        """ 
        calculate_energy_consumption reads the energy measurements from the correct csv, averages it and then integrates it over inference time 

        :param board_snr: string containing information the board snr
        :param data_dir: directory path containing result.json and power_measurements_<board_snr>.csv of the board

        :return: None. Writes energy consumption and mean power consumption to results.json
        """ 

        # get paths FIXME: make paths more robust for e.g. Windows 
        power_measurement_file_name = "power_measurements_" + board_snr +".csv"
        csv_path = data_dir + "/" + power_measurement_file_name
        results_path = data_dir + "/" + "results.json"

        try:
            # load results from json
            with open(results_path) as f:
                results = json.loads(f.read())
        except FileNotFoundError as e:
            raise NotImplementedError("Not implemented proper handling if result does not exist. should actually not be the case and not be ignored")
        except Exception as e:
            raise NotImplementedError("proper error handling")

        try:
            data = pd.read_csv(csv_path)
            # get all power consumption measurements
            values = np.asarray(data["Power Consumption"])

            threshold = self.params["power_measurement_threshold"]
            start = None
            end = None
            for val in values:
                if start is None and val > threshold:
                    start = int(np.where(val == values)[0])
                elif start is not None and val < threshold:
                    end = int(np.where(val == values)[0])
                    break

            # the value with the highest gradient is
            mean_power_consumption = np.mean(values[start:end])  # measured in uA
            mean_power_consumption = mean_power_consumption * (10 ** -6)  # in A

            voltage = 3.3  # in V

            # get inference time from board
            try:
                inf_time = self.get_inference_information_from_results(board_snr,results) # in ms
            except ValueError as e:
                inf_time = 0

            # convert to seconds
            inf_time = inf_time * (10 ** -3)  # in s

            # calculate energy by Energy = Voltage x Current x time
            energy_consumption = voltage * mean_power_consumption * inf_time  # in J
            energy_consumption = energy_consumption * (10 ** 3)  # in mJ

            # save energy consumption to results
            results["energy_information"] = self.set_result_value_for_board(board_snr, "energy_information", float(energy_consumption), results)
            results["mean_power_information"] = self.set_result_value_for_board(board_snr, "mean_power_information", float(mean_power_consumption),results)

        except Exception as e:
            results["energy_information"] = self.set_result_value_for_board(board_snr, "energy", str(e), results)

        # save to results.json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)


    def evaluate_energy_consumption_and_inference_speed(self):
        """ Evaluate all preselected models on the MCU. """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        for idx, individual in enumerate(self.preselected_individuals):
            print(f"Evaluate energy of {individual} (index: {idx})")

            # flash tflite model on individual board
            if len(self.params["boards"]) > 0:       
                for board in self.params["boards"]:
                    tflite_path = "../" +path + individual + "/models/model_tflite_untrained.tflite"
                    cpp_path = '../tflite/evonas_tflite/src/model.cpp'
                    flasher_path = './tools/flash_tflite_model.sh'

                    # start measuring
                    ppk2 = init_ppk2(board["ppk"])
                    time.sleep(1)
                    subprocess.call(['bash', '-i',flasher_path, tflite_path, cpp_path, board["model"], board["snr"]])
                    time.sleep(1)

                    # if no ppk connected, measuring the power consumption is not possible
                    if ppk2 is not None:
                        stop_measuring(ppk2)

                        # start measuring energy consumption
                        args = ['python tools/measure_power_consumption.py', path + individual, board["snr"], board["ppk"], f'{self.params["power_measurement_nb_samples_average"]}']
                        command = " ".join(args) # joining args separated by space
                        proc_energy = Popen(command, shell=True)

                    # get inference time from Serial port
                    args = ['python tools/measure_inference_time.py', path + individual, board["model"], board["snr"]]
                    command = " ".join(args) # joining args separated by space
                    proc_inference = Popen(command, shell=True)

                    # wait for inference time measurement to finish
                    proc_inference.wait()

                    # if no ppk connected, measuring the power consumption is not possible
                    if ppk2 is not None:
                        # wait for energy consumption measurement to finish
                        try:
                            proc_energy.wait(timeout=30)

                            # calculate energy consumption
                            self.calculate_energy_consumption(board["snr"], path + individual)
                        except:
                            pass

            else: # no boards 
                raise ValueError(f'No boards are set. Length of params["boards"]: {len(self.params["boards"])}')
          

            time.sleep(2)

    def selection(self):
        # calculate fitness of all preselected models
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        models_with_fitness = dict()
        for individual in self.preselected_individuals:
            with open(path + individual + '/results.json', 'r') as f:
                results = json.loads(f.read())
                fitness = calculate_fitness(results, self.params)

                # save fitness together with the individual name to sort and select them later
                models_with_fitness[f'{individual}'] = fitness

            # save fitness in results.json
            with open(path + individual + '/results.json', 'w') as f:
                results['fitness'] = float(fitness)
                json.dump(results, f, indent=2)

        # sort them by their achieved fitness
        models_with_fitness = sorted(models_with_fitness.items(), key=lambda item: item[1], reverse=True)

        # save the model with the best fitness to find it easier later on
        self.my_saver.save_best_individual(self.generation_counter, models_with_fitness[0])

        self.best_models_current_generation = \
            [models_with_fitness[i][0] for i in range(self.params['nb_best_models_crossover']) if
             i < len(models_with_fitness)]

    def crossover(self):
        """ Crossover the best chromosomes to get the population for the next generation. """
        self.population_next_generation, self.parents_names = self.my_gene_pool.crossover(
            path=f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/',
            fittest_chromosomes=self.best_models_current_generation)

    def mutation(self):
        """ Mutation of the population previously generated by crossover. """
        self.population_genotype = []
        for chromosome in self.population_next_generation:
            mutated_chromosome = self.my_gene_pool.mutate_chromosome(chromosome)
            self.population_genotype.append(mutated_chromosome)

    def _process_model_translation(self, chromosome):
        try:
            model = translate(chromosome, self.params['input_shape'], self.params['nb_classes'],
                                self.params['sample_rate'])
        except:
            raise ValueError(f"Error when translating from genotype to phenotype. Chromosome: {chromosome}")    
        try:
            model = substitute_tflite_layer(model, self.params["input_shape"])
        except:
            raise ValueError(f"Error when substituting STFT and MAG layers. Chromosome: {chromosome}")
        return model  

    def _process_model_conversion(self, model):
        try:
            tflite_model = convert_to_tflite(model, np.random.uniform(size=(200, self.params["input_shape"][0],
                                                                                 self.params["input_shape"][1])))
        except:
            raise ValueError(f"Error when converting to TFLite")
        
        return tflite_model

    def prepare_generation(self, current_generation: int):
        self.individuals_names = []
        self.population_phenotype = []
        self.population_phenotype_tflite = []
        self.preselected_individuals = []
        self.best_models_current_generation = []
        self.population_next_generation = []

        self._generate_individuals_names()
        if self.parents_names is not None:
            self.my_saver.save_parents(self.generation_counter, self.individuals_names, self.parents_names)
            self.parents_names = None

        self.generation_counter = current_generation

        # convert chromosomes to models and to tflite models in parallel
        with get_context("spawn").Pool(os.cpu_count()) as pool:
            models = pool.map(self._process_model_translation, self.population_genotype)
            # translate all chromosomes: genotype (chromosome) --> phenotype (tf.keras.Model)
            tflite_models = pool.map(self._process_model_conversion, models)

        # save models 
        for model in models:
            self.population_phenotype.append(model)

        for tflite_model in tflite_models:
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
