import os.path
from subprocess import Popen

import pandas as pd
from coolname import generate_slug
import json
import numpy as np
import time
import subprocess
import nvidia_smi
from tqdm import tqdm
import copy
from omegaconf import DictConfig

from .src.genepool import GenePool
from .src.translation import translate
from utils.saver import Saver
from utils.loader import Loader
from .src.objective_function import calculate_fitness

from .utils.convert_to_tflite import convert_to_tflite
from .utils.substitute_tflite_layer import substitute_tflite_layer
from .utils.save_ram_rom_usage import save_ram_rom_usage
from tools.measure_power_consumption import init_ppk2, stop_measuring
from multiprocessing import get_context

from multiprocessing import Process
import time


class GeneticAlgorithm:
    def __init__(self, cfg: DictConfig, saver: Saver, loader: Loader = None):
        self.cfg = cfg
        self.my_saver = saver
        self.my_gene_pool = GenePool(cfg)

        # define all variables that change after each generation
        if loader is None:
            self.individuals: dict = {}
        else:
            self.individuals: dict = loader.load_individuals()

        self.generation_counter: int = 0  # information in which generation we are currently

    def init_first_generation(self):
        # update parameters for the first generation
        self.update_population_size()
        self.update_num_best_models_crossover()
        self.update_mutation_rate()

        # generate random names for all individuals
        self.individuals = self._generate_population_names()

        # create random chromosomes for all individuals
        for name in self.individuals.keys():
            random_chromosome = self.my_gene_pool.create_gene_sequence()
            self.individuals[name]["genotype"] = random_chromosome

    def create_population_first_generation(self):
        random_chromosome = self.my_gene_pool.create_gene_sequence()
        raise NotImplementedError("create_population_first_generation is not implemented yet")

        # TODO: implement this function


    def prepare_generation(self, current_generation: int):
        # increment generation counter
        self.generation_counter = current_generation

        # apply decays to hyperparameters to go away from exploration to exploitation
        self.update_population_size()
        self.update_num_best_models_crossover()
        self.update_mutation_rate()

        # create generation dir
        self.my_saver.create_generation_dir(self.individuals, self.generation_counter)

        # save population genotype
        self.my_saver.save_population_genotype(self.individuals, self.generation_counter)

        # convert chromosomes to models and to tflite models in parallel
        cpus = os.cpu_count() - 4
        # cpus = 8
        with get_context("spawn").Pool(cpus) as pool:
            pool.map(self._process_model_translation_and_conversion, self.individuals)

        print("Finished translating and converting models")
    
    def _generate_random_name(self):
        return generate_slug(2).replace("-", "_") + f"_{self.generation_counter + 1}"

    def _generate_population_names(self):
        names = set()
        population_size = self.cfg.hyperparameters.population_size.value
        while len(names) < population_size:
            random_name = self._generate_random_name()
            if random_name not in names:
                names.add(random_name)

        names_dict = {name: {} for name in sorted(names)}
        return names_dict
    

    def evaluate_memory_footprint(self):
        """ 
        Load memory footprint after converting the model to TFLite.
        Determine models (dependent on the thresholds specified in main.py) that will be further evaluated. 
        """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        individuals_copy = copy.deepcopy(self.individuals)
        for individual in individuals_copy.keys():
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

            with open(path + individual + '/results.json', 'w') as f:
                json.dump(d, f, indent=2)

        if len(self.individuals) == 0:
            raise Exception("All models are too big in terms of file size. Therefore none of the generated models will"
                            " be further evaluated. Think about adjusting your GA parameters.")

    def train_neural_networks(self):
        # train all neural networks
        # --> start several training processes here
        min_free_space = self.cfg.hyperparameters.min_free_space_gpu.value
        procs = []
        individuals_names = list(self.individuals.keys())
        tqdm_bar = tqdm(total=len(individuals_names), desc="Training models")

        idx = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        
        while idx < len(individuals_names):
            procs = [p for p in procs if p.is_alive()]
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            if info.free > min_free_space and len(procs) < 4:
                command = 'python neural_architecture_search/src/train.py ' + \
                          f'--results_dir {self.my_saver.results_dir} ' + \
                          f'--gen_dir Generation_{self.generation_counter} ' + \
                          f'--individual_dir {individuals_names[idx]} ' + \
                          f'--dataset {self.cfg.hyperparameters.dataset_name.value} ' + \
                          f'--num_epochs {self.cfg.hyperparameters.num_epochs.value} ' + \
                          f'--batch_size {self.cfg.hyperparameters.batch_size.value} ' + \
                          f'--loss {self.cfg.hyperparameters.loss.value} ' +\
                          f'--metrics {" ".join(str(i) for i in self.cfg.hyperparameters.metrics.value)} ' + \
                          f'--optimizer {self.cfg.hyperparameters.optimizer.value} ' 
                proc = Process(target=lambda: Popen(command, shell=True).wait())
                proc.start()
                procs.append(proc)
                idx += 1
                tqdm_bar.update(1)
            time.sleep(10)

        nvidia_smi.nvmlShutdown()
        
        # make sure to wait until all processes are finished
        for p in procs:
            try:
                p.join(timeout=300)
            except:
                p.join()


    @staticmethod
    def get_inference_information_from_results(board_snr, results):
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

    @staticmethod
    def set_result_value_for_board(board_snr, category, value, results):
        """ 
        update_or_set_result_value checks if the board information is already available and appends it to the dict
        structure of "<type>_information" is expected to be this:
        "<type>_information" : {
                    "board_1" : [board_1_information],
                    "board_2: : [board_2_information],
                    ....
                    "board_n" : [board_n_information]
        }
        :param board_snr: string containing information the board snr
        :param category: string that specifies the type of results key. e.g. "energy_information" or "mean_power_information"
        :param value: the value that will be added to the information for the category
        :param results: the current data that has already been saved

        :raises: RuntimeError if the category with board_snr is already set
        :return: dictionary with updated information
        """

        # board snr should be unique id that serves as key for the information
        _id = board_snr

        # get previous category information if exists
        information = {}
        if category in results:

            # get old information of category
            information = results[category]

            # inference information of specified board should not already be contained beforehand
            if _id in information:
                raise RuntimeError(f"result.json already contains {category} of the specified board {board_snr}")

        # append new information for board with id
        information[_id] = value

        # return the updated information
        return information

    def calculate_energy_consumption(self, board_snr: str, power_measurement_threshold: int, data_dir: str):
        """ 
        calculate_energy_consumption reads the energy measurements from the correct csv, averages it and then integrates it over inference time 

        :param board_snr: string containing information the board snr
        :param data_dir: directory path containing result.json and power_measurements_<board_snr>.csv of the board

        :return: None. Writes energy consumption and mean power consumption to results.json
        """

        # get paths FIXME: make paths more robust for e.g. Windows 
        power_measurement_file_name = "power_measurements_" + board_snr + ".csv"
        csv_path = data_dir + "/" + power_measurement_file_name
        results_path = data_dir + "/" + "results.json"

        try:
            # load results from json
            with open(results_path) as f:
                results = json.loads(f.read())
        except FileNotFoundError as e:
            raise NotImplementedError(
                "Not implemented proper handling if result does not exist. "
                "Should actually not be the case and not be ignored")
        except Exception as e:
            raise NotImplementedError("proper error handling")

        try:
            data = pd.read_csv(csv_path)
            # get all power consumption measurements
            values = np.asarray(data["Power Consumption"])

            # omit the first 10k values as they are not stable
            values = values[10000:]

            values_averaged = pd.Series(values).rolling(self.cfg.hyperparameters.power_measurement_num_samples_average.value).mean()

            start = np.where(values_averaged > power_measurement_threshold)[0][0]
            end = np.where(values_averaged < power_measurement_threshold)[0][np.where(values_averaged < power_measurement_threshold)[0] > np.where(values_averaged > power_measurement_threshold)[0][0]][0]

            # the value with the highest gradient is
            mean_power_consumption = np.mean(values[start:end])  # measured in uA

            mean_power_consumption = mean_power_consumption * (10 ** -6)  # in A

            voltage = 3.3  # in V

            # get inference time from board
            inf_time = self.get_inference_information_from_results(board_snr, results)  # in ms

            # convert to seconds
            inf_time = inf_time * (10 ** -3)  # in s

            # calculate energy by Energy = Voltage x Current x time
            energy_consumption = voltage * mean_power_consumption * inf_time  # in J
            energy_consumption = energy_consumption * (10 ** 3)  # in mJ
            
            # save energy consumption to results
            results["energy_information"] = self.set_result_value_for_board(board_snr, "energy_information",
                                                                            float(energy_consumption), results)
            results["mean_power_information"] = self.set_result_value_for_board(board_snr, "mean_power_information",
                                                                                float(mean_power_consumption), results)

        except Exception as e:
            results["energy_information"] = self.set_result_value_for_board(board_snr, "energy", str(e), results)

        # save to results.json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def evaluate_energy_consumption_and_inference_speed(self):
        """ Evaluate all preselected models on the MCU. """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        individuals_names = list(self.individuals.keys())
        for idx, individual in tqdm(enumerate(individuals_names), total=len(individuals_names)):
            print(f"Evaluate energy of {individual} (index: {idx+1})")

            # error log 
            error_log_path = path + individual + '/error_log.txt'

            # flash tflite model on individual board
            if len(self.cfg.boards.value) > 0:
                for board in self.cfg.boards.value:
                    tflite_path = path + individual + '/models/model_tflite_untrained.tflite'
                    cpp_path = '../tflite/evonas_tflite/src/model.cpp'
                    flasher_path = './tools/flash_tflite_model.sh'

                    # init PPK2 --> THIS NEEDS TO BE DONE BEFORE FLASHING THE MODEL (would not work otherwise)
                    ppk2 = init_ppk2(board.ppk)
                    time.sleep(2)  # --> important to wait a bit before flashing the model

                    # flash tflite model on board
                    try:
                        ret_val = subprocess.call(['bash', '-i', flasher_path, tflite_path, cpp_path, board.model, board.snr])
                    except Exception as e:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error when flashing model on board {board.snr} - exception: {str(e)}.\n")
                    
                    if ret_val != 0:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error when flashing model on board {board.snr}. Ret val: {ret_val}.\n")
                        
                        # disconnect ppk2 
                        del ppk2
                        time.sleep(3)
                        continue
                    
                    # wait for the board to boot
                    time.sleep(5)

                    # save RAM and ROM usage to results.json (available after the project was built)
                    save_ram_rom_usage("tflite/build-" + board.model, path + individual + "/" + "results.json")

                    # if no ppk connected, measuring the power consumption is not possible
                    proc_energy = None
                    if ppk2 is not None:
                        del ppk2
                        time.sleep(3)

                        # start measuring energy consumption
                        args = ['python tools/measure_power_consumption.py', path + individual, board.snr,
                                board.ppk, f'{self.cfg.hyperparameters.power_measurement_num_samples_average.value}']
                        command = " ".join(args)  # joining args separated by space
                        proc_energy = Popen(command, shell=True)

                        time.sleep(2)

                    # wait for inference time measurement to finish
                    try:
                        # get inference time from Serial port
                        args = ['python tools/measure_inference_time.py', path + individual, board.model, board.snr]
                        command = " ".join(args)  # joining args separated by space
                        proc_inference = Popen(command, shell=True)

                        proc_inference.wait()
                    except Exception as e:
                        # save error log
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error when measuring inference time on board {board.snr}.\n Exception: {str(e)}\n")

                    # if no ppk connected, measuring the power consumption is not possible
                    if proc_energy is not None:
                        # wait for energy consumption measurement to finish
                        try:
                            proc_energy.wait(timeout=10)
                        except Exception as e:
                            # save error log
                            with open(error_log_path, 'a') as f:
                                f.write(f"Error when measuring energy consumption on board {board.snr}.\n")

                        try:
                            # calculate energy consumption
                            self.calculate_energy_consumption(board.snr, board.power_measurement_threshold, path + individual)
                        except Exception as e:
                            # save error log
                            with open(error_log_path, 'a') as f:
                                f.write(f"Error when calculating energy consumption on board {board.snr}.\n")
                                f.write(f"Exception: {str(e)}\n")
                    time.sleep(3)
            else:  # no boards
                raise ValueError(f'No boards are set. Length of params["boards"]: {len(self.cfg.boards.value)}')
            

    def selection(self):
        # calculate fitness of all preselected models
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        individuals_names = list(self.individuals.keys())
        for individual in individuals_names:
            with open(path + individual + '/results.json', 'r') as f:
                results = json.loads(f.read())
                fitness, error = calculate_fitness(results, self.cfg)

                self.individuals[individual]["fitness"] = fitness

            # save fitness in results.json
            with open(path + individual + '/results.json', 'w') as f:
                results['fitness'] = float(fitness)
                results['error'] = str(error)
                json.dump(results, f, indent=2)

        # sort individuals by their achieved fitness
        self.individuals = dict(sorted(self.individuals.items(), key=lambda item: item[1]["fitness"], reverse=True))

        # save the model with the best fitness to find it easier later on
        best_individual_name = list(self.individuals.keys())[0]
        best_individual_fitness = self.individuals[best_individual_name]["fitness"]
        self.my_saver.save_best_individual(self.generation_counter, best_individual_name, best_individual_fitness)

        # omit the individuals that are not in the top x
        self.individuals = dict(list(self.individuals.items())[:self.cfg.hyperparameters.num_best_models_crossover.value])

    def crossover(self):
        """ Crossover the best chromosomes to get the population for the next generation. """
        population_next_generation, parents_names = self.my_gene_pool.crossover(
            path=f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/',
            fittest_chromosomes=list(self.individuals.keys()))

        # create names for next generation
        self.individuals = self._generate_population_names()

        # add the new individuals to the dict
        for individual, chromosome in zip(self.individuals.keys(), population_next_generation):
            self.individuals[individual]["genotype"] = chromosome

        # save parents
        self.my_saver.save_parents(self.generation_counter, list(self.individuals.keys()), parents_names)

    def update_population_size(self):
        """
        Apply population size decay after each generation.
        """
        decay = self.cfg.hyperparameters.population_size_decay.value
        self.cfg.hyperparameters.population_size.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def update_num_best_models_crossover(self):
        """ 
        Apply num_best_models_crossover decay after each generation.
        """
        decay = self.cfg.hyperparameters.num_best_models_crossover_decay.value
        self.cfg.hyperparameters.num_best_models_crossover.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def update_mutation_rate(self):
        """ 
        Apply mutation rate decay after each generation.
        """
        decay = self.cfg.hyperparameters.mutation_rate_decay.value
        self.cfg.hyperparameters.mutation_rate.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def mutation(self):
        """ Mutation of the population previously generated by crossover. """
        for name in self.individuals.keys():
            chromosome = self.individuals[name]["genotype"]
            mutated_chromosome = self.my_gene_pool.mutate_chromosome(chromosome)
            self.individuals[name]["genotype"] = mutated_chromosome

    def _process_model_translation_and_conversion(self, individual_name: str):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print("Could not set memory growth in function _process_model_translation")
            pass
            
        # translate chromosome to TensorFlow model
        try:
            model = translate(self.individuals[individual_name]['genotype'], 
                              self.cfg.hyperparameters.input_shape.value, 
                              self.cfg.hyperparameters.num_classes.value, 
                              self.cfg.hyperparameters.top_activation.value, 
                              self.cfg.hyperparameters.sample_rate.value)
        except:
            raise ValueError(f"Error when translating from genotype to phenotype. Chromosome: {self.individuals[individual_name]['genotype']}")
        
        # save TensorFlow model
        self.my_saver.save_population_phenotype(individual_name, self.generation_counter, model)

        # substitute STFT and MAG layers
        try:
            model_substituted = substitute_tflite_layer(model, self.cfg.hyperparameters.input_shape.value)
        except:
            raise ValueError(f"Error when substituting STFT and MAG layers.")

        try:
            # generate dummy data for quantization
            if len(self.cfg.hyperparameters.input_shape.value) == 3:
                representative_dataset = np.random.uniform(size=(200, self.cfg.hyperparameters.input_shape.value[0], self.cfg.hyperparameters.input_shape.value[1], self.cfg.hyperparameters.input_shape.value[2]))
            else:
                representative_dataset = np.random.uniform(size=(200, self.cfg.hyperparameters.input_shape.value[0], self.cfg.hyperparameters.input_shape.value[1]))

            tflite_model = convert_to_tflite(model_substituted, representative_dataset)
        except:
            raise ValueError(f"Error when converting to TFLite")

        print(f"Save TFLite model of {individual_name}")
        self.my_saver.save_population_phenotype_tflite(individual_name, self.generation_counter, tflite_model)


