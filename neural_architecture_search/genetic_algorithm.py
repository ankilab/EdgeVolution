import logging
import os.path
from subprocess import Popen

import pandas as pd
from coolname import generate_slug
import json
import numpy as np
import time
import subprocess
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
from multiprocessing import get_context
from multiprocessing import Process

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    def __init__(self, cfg: DictConfig, saver: Saver, loader: Loader = None,
                 surrogate=None, search_space_registry=None):
        self.cfg = cfg
        self.my_saver = saver
        self.my_gene_pool = GenePool(cfg)
        self.profiling_stats = profiling_stats  # Optional profiling statistics

        # define all variables that change after each generation
        if loader is None:
            self.individuals: dict = {}
        else:
            self.individuals: dict = loader.load_individuals()

        self.generation_counter: int = 0  # information in which generation we are currently

        # Surrogate model for accuracy prediction
        self.surrogate = surrogate
        self.search_space_registry = search_space_registry
        self._skipped_individuals: dict = {}

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

    def prepare_generation(self, current_generation: int):
        """
        Prepare the generation by applying decays to hyperparameters and creating the generation directory.
        Convert the chromosomes to models and to tflite models in parallel.

        :param current_generation: int, the current generation number

        :return: None
        """
        # increment generation counter
        self.generation_counter = current_generation

        print(f"\nPreparing generation {self.generation_counter}...")
        print(f"  Population size: {self.cfg.hyperparameters.population_size.value}")

        # apply decays to hyperparameters to go away from exploration to exploitation
        self.update_population_size()
        self.update_num_best_models_crossover()
        self.update_mutation_rate()

        # create generation dir
        self.my_saver.create_generation_dir(self.individuals, self.generation_counter)

        # save population genotype
        self.my_saver.save_population_genotype(self.individuals, self.generation_counter)

        # Track time for parallel translation and conversion
        translation_conversion_start = time.time()
        
        print(f"  Translating {len(self.individuals)} chromosomes to models and converting to TFLite...")
        
        # convert chromosomes to models and to tflite models in parallel
        # I don't want to use the whole CPU power for this task to avoid freezing the system etc.
        cpus = os.cpu_count() - 4

        # fix for cpus < 1
        if cpus < 1:
            cpus = 1

        with get_context("spawn").Pool(cpus) as pool:
            pool.map(self._process_model_translation_and_conversion, self.individuals)

        logger.info("Finished translating and converting models")
    
    def _generate_random_name(self):
        """
        Generate a random name for the individual.

        :return: str, the random name
        """
        return generate_slug(2).replace("-", "_") + f"_{self.generation_counter + 1}"

    def _generate_population_names(self):
        """
        Generate random names for the population.

        :return: dict, the dictionary with the random names
        """
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

        :return: None. Writes memory footprint to results.json
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
        """"
        Train all preselected models on the MCU.

        :return: None. Writes training results to results.json
        """
        procs = []
        individuals_names = list(self.individuals.keys())
        tqdm_bar = tqdm(total=len(individuals_names), desc="Training models")
        idx = 0

        # Auto-detect GPU and compute parallelization settings
        gpu_available = False
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total_memory = info.total
            total_gpu_gb = total_memory / (1024 ** 3)
            min_free_threshold = max(int(total_memory * 0.15), 500_000_000)
            max_parallel = min(max(1, int(total_gpu_gb / 2)), 8)
            gpu_available = True
            logger.info(f"GPU detected: {total_gpu_gb:.1f} GB total, "
                        f"free-memory threshold: {min_free_threshold / 1e6:.0f} MB, "
                        f"max parallel processes: {max_parallel}")
        except Exception:
            max_parallel = 1
            logger.info("No GPU detected or nvidia_smi unavailable. "
                        "Training sequentially (1 process at a time).")

        # start training processes
        while idx < len(individuals_names):
            procs = [p for p in procs if p.is_alive()]

            can_launch = False
            if gpu_available:
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                if info.free > min_free_threshold and len(procs) < max_parallel:
                    can_launch = True
            else:
                if len(procs) < max_parallel:
                    can_launch = True

            if can_launch:
                command = 'python3 neural_architecture_search/src/train.py ' + \
                          f'--results_dir {self.my_saver.results_dir} ' + \
                          f'--gen_dir Generation_{self.generation_counter} ' + \
                          f'--individual_dir {individual_name} ' + \
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

        if gpu_available:
            nvidia_smi.nvmlShutdown()

        # make sure to wait until all processes are finished
        for p in procs:
            try:
                p.join(timeout=300)
            except Exception:
                p.join()
        
        # Record training times for profiling
        if self.profiling_stats is not None:
            for individual_name in individuals_names:
                if individual_name in training_start_times:
                    training_duration = time.time() - training_start_times[individual_name]
                    self.profiling_stats.record_model_operation(
                        self.generation_counter, individual_name, "training", training_duration
                    )
        
        print(f"\n{'='*80}")
        print(f"Training completed for generation {self.generation_counter}")
        print(f"{'='*80}\n")


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

        # get paths
        power_measurement_file_name = "power_measurements_" + board_snr + ".csv"
        csv_path = os.path.join(data_dir, power_measurement_file_name)
        results_path = os.path.join(data_dir, "results.json")

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

            # omit the first 10k values as they are not stable yet (initialization of the current measurements)
            values = values[10000:]

            values_averaged = pd.Series(values).rolling(self.cfg.hyperparameters.power_measurement_num_samples_average.value).mean()

            # Auto-detect inference window from power trace.
            idle_region = values_averaged[:5000]
            idle_mean = np.nanmean(idle_region)
            idle_std = np.nanstd(idle_region)
            adaptive_threshold = idle_mean + max(3 * idle_std, 1000)

            above = np.where(values_averaged > adaptive_threshold)[0]
            if len(above) > 0:
                start = above[0]
                below_after_start = np.where(values_averaged[start:] < adaptive_threshold)[0]
                if len(below_after_start) > 0:
                    end = start + below_after_start[0]
                else:
                    end = len(values_averaged)
            else:
                start = 0
                end = len(values_averaged)

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
        """ 
        Evaluate all preselected models on the MCU.
        
        :return: None. Writes energy consumption and inference speed to results.json
        """
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        from tools.measure_power_consumption import init_ppk2, stop_measuring

        individuals_names = list(self.individuals.keys())
        print(f"\n{'='*80}")
        print(f"Starting MCU evaluation for {len(individuals_names)} models")
        print(f"{'='*80}\n")
        
        # Track deployment times for all models
        deployment_times = {}
        
        for idx, individual in tqdm(enumerate(individuals_names), total=len(individuals_names)):
            logger.info(f"Evaluate energy of {individual} (index: {idx+1})")

            # error log
            error_log_path = path + individual + '/error_log.txt'

            # flash tflite model on individual board
            if len(self.cfg.boards.value) > 0:
                for board in self.cfg.boards.value:
                    # Use absolute paths so they resolve correctly after cd in flash script
                    tflite_path = os.path.abspath(path + individual + '/models/model_tflite_untrained.tflite')
                    cpp_path = os.path.abspath('tflite/edgevolution_tflite/src/model.cpp')
                    flasher_path = os.path.abspath('tools/flash_tflite_model.sh')
                    results_path = path + individual + '/results.json'

                    # init PPK2 --> THIS NEEDS TO BE DONE BEFORE FLASHING THE MODEL (would not work otherwise)
                    ppk2 = init_ppk2(board.ppk)
                    time.sleep(2)  # --> important to wait a bit before flashing the model

                    # flash tflite model on board
                    ret_val = -1
                    try:
                        ret_val = subprocess.call(['bash', flasher_path, tflite_path, cpp_path, board.model, board.snr])
                    except Exception as e:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error when flashing model on board {board.snr} - exception: {str(e)}.\n")

                    if ret_val != 0:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error when flashing model on board {board.snr}. Ret val: {ret_val}.\n")
                        # Write flash error to results.json so calculate_fitness sees it
                        try:
                            with open(results_path) as f:
                                results = json.loads(f.read())
                            results['flash_error'] = f"Flash failed on board {board.snr} (ret_val={ret_val})"
                            with open(results_path, 'w') as f:
                                json.dump(results, f, indent=2)
                        except Exception:
                            pass
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
                        args = ['python3 tools/measure_power_consumption.py', path + individual, board.snr,
                                board.ppk, f'{self.cfg.hyperparameters.power_measurement_num_samples_average.value}']
                        command = " ".join(args)  # joining args separated by space
                        proc_energy = Popen(command, shell=True)

                        time.sleep(2)

                    # wait for inference time measurement to finish
                    measurement_start = time.time()
                    try:
                        # get inference time from Serial port
                        args = ['python3 tools/measure_inference_time.py', path + individual, board.model, board.snr]
                        command = " ".join(args)  # joining args separated by space
                        proc_inference = Popen(command, shell=True)

                        proc_inference.wait()
                        measurement_duration = time.time() - measurement_start
                        print(f"  Inference measurement took {measurement_duration:.2f} seconds")
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
            
            # Record total deployment time for this individual
            individual_deployment_duration = time.time() - individual_deployment_start
            deployment_times[individual] = individual_deployment_duration
            print(f"  Total MCU evaluation for {individual}: {individual_deployment_duration:.2f} seconds")
        
        # Save deployment times to a file so they can be loaded by the main process
        deployment_times_file = path + '../deployment_times.json'
        with open(deployment_times_file, 'w') as f:
            json.dump({
                'generation': self.generation_counter,
                'deployment_times': deployment_times
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"MCU evaluation completed for generation {self.generation_counter}")
        print(f"{'='*80}\n")
            

    def surrogate_prescreen(self):
        """
        Use the surrogate model to pre-screen individuals and skip training
        for those confidently predicted to perform poorly.

        Skipped individuals get their predicted val_acc written to results.json
        so that selection() works unchanged.
        """
        if self.surrogate is None or self.search_space_registry is None:
            return

        to_train, to_skip, _ = self.surrogate.prescreen(
            self.individuals, self.search_space_registry.encode
        )

        if not to_skip:
            self._skipped_individuals = {}
            return

        predictions = self.surrogate.get_predictions()
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        # Write surrogate-predicted results for skipped individuals
        self._skipped_individuals = {}
        for name in to_skip:
            self._skipped_individuals[name] = self.individuals[name]
            pred = predictions.get(name, {})
            predicted_acc = pred.get("predicted_acc", 0.0)

            results_path = path + name + '/results.json'
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                results = {}

            results['val_acc'] = predicted_acc
            results['surrogate_skipped'] = True

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        # Remove skipped individuals so train_neural_networks skips them
        self.individuals = {
            name: data for name, data in self.individuals.items()
            if name in to_train
        }

    def collect_surrogate_data(self):
        """
        After training, collect actual accuracies, update the surrogate model,
        and merge skipped individuals back into self.individuals.
        """
        if self.surrogate is None or self.search_space_registry is None:
            return

        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'
        predictions = self.surrogate.get_predictions()

        # Collect observations from actually-trained individuals
        for name, data in self.individuals.items():
            results_path = path + name + '/results.json'
            if not os.path.exists(results_path):
                continue
            with open(results_path, 'r') as f:
                results = json.load(f)
            val_acc = results.get('val_acc')
            if val_acc is not None:
                encoding = self.search_space_registry.encode(data["genotype"])
                self.surrogate.add_observation(encoding, float(val_acc))

        # Build per-individual records for logging
        individual_records = []
        all_names = list(self.individuals.keys()) + list(self._skipped_individuals.keys())

        for name in all_names:
            pred = predictions.get(name, {})
            predicted_acc = pred.get("predicted_acc")
            uncertainty = pred.get("uncertainty")
            skipped = name in self._skipped_individuals

            # Read actual accuracy
            results_path = path + name + '/results.json'
            actual_acc = None
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                actual_acc = results.get('val_acc')

            individual_records.append({
                "name": name,
                "predicted_acc": predicted_acc,
                "uncertainty": uncertainty,
                "actual_acc": float(actual_acc) if actual_acc is not None else None,
                "skipped": skipped,
            })

        # Merge skipped individuals back
        self.individuals.update(self._skipped_individuals)
        self._skipped_individuals = {}

        # Log generation data
        surrogate_dir = f'{self.my_saver.results_dir}/surrogate'
        self.surrogate.log_generation(
            self.generation_counter, individual_records, surrogate_dir
        )

        # Fit and save if enough data
        if self.surrogate.is_ready:
            self.surrogate.fit()
            self.surrogate.save(surrogate_dir)

    def selection(self):
        """
        Select the best individuals based on their fitness.
        
        :return: None. Writes fitness to results.json
        """
        print(f"\nCalculating fitness for {len(self.individuals)} models...")
        
        # calculate fitness of all preselected models
        path = f'{self.my_saver.results_dir}/Generation_{self.generation_counter}/'

        individuals_names = list(self.individuals.keys())
        for individual in individuals_names:
            with open(path + individual + '/results.json', 'r') as f:
                results = json.loads(f.read())

                # use the results to calculate the fitness (config is needed for the fitness calculation
                # as it contains the weighting values)
                fitness, error = calculate_fitness(results, self.cfg)

                # Save calculated fitness for each individual
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
        
        print(f"  Best individual: {best_individual_name} (fitness: {best_individual_fitness:.4f})")

        # omit the individuals that are not in the top x
        num_selected = self.cfg.hyperparameters.num_best_models_crossover.value
        self.individuals = dict(list(self.individuals.items())[:num_selected])
        print(f"  Selected top {num_selected} individuals for next generation")

    def crossover(self):
        """ 
        Crossover the best chromosomes to get the population for the next generation. 
        
        :return: None. Writes parents to results.json
        """
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

        :return: None
        """
        decay = self.cfg.hyperparameters.population_size_decay.value
        self.cfg.hyperparameters.population_size.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def update_num_best_models_crossover(self):
        """ 
        Apply num_best_models_crossover decay after each generation.

        :return: None
        """
        decay = self.cfg.hyperparameters.num_best_models_crossover_decay.value
        self.cfg.hyperparameters.num_best_models_crossover.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def update_mutation_rate(self):
        """ 
        Apply mutation rate decay after each generation.

        :return: None
        """
        decay = self.cfg.hyperparameters.mutation_rate_decay.value
        self.cfg.hyperparameters.mutation_rate.value = next(sublist[1] for sublist in decay[::-1] if self.generation_counter+1 >= sublist[0])

    def mutation(self):
        """ 
        Mutation of the population previously generated by crossover. 
        
        :return: None
        """
        for name in self.individuals.keys():
            chromosome = self.individuals[name]["genotype"]
            mutated_chromosome = self.my_gene_pool.mutate_chromosome(chromosome)
            self.individuals[name]["genotype"] = mutated_chromosome

    def _process_model_translation_and_conversion(self, individual_name: str):
        """
        Translate the chromosome to a TensorFlow model and convert it to a TFLite model.
        
        :param individual_name: str, the name of the individual
        
        :return: None
        """
        translation_start_time = time.time()
        
        try:
            # set memory growth for GPU
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass  # GPU memory growth may already be set or no GPU available

        # translate chromosome to TensorFlow model
        try:
            model = translate(self.individuals[individual_name]['genotype'],
                              self.cfg.hyperparameters.input_shape.value,
                              self.cfg.hyperparameters.num_classes.value,
                              self.cfg.hyperparameters.top_activation.value,
                              self.cfg.hyperparameters.sample_rate.value)
        except Exception as e:
            raise ValueError(f"Error when translating from genotype to phenotype. Chromosome: {self.individuals[individual_name]['genotype']}") from e


        # save TensorFlow model
        self.my_saver.save_population_phenotype(individual_name, self.generation_counter, model)

        # substitute STFT and MAG layers
        try:
            model_substituted = substitute_tflite_layer(model, self.cfg.hyperparameters.input_shape.value)
        except Exception as e:
            raise ValueError("Error when substituting STFT and MAG layers.") from e

        conversion_start_time = time.time()
        
        try:
            # generate dummy data for quantization
            if len(self.cfg.hyperparameters.input_shape.value) == 3:
                representative_dataset = np.random.uniform(size=(200, self.cfg.hyperparameters.input_shape.value[0], self.cfg.hyperparameters.input_shape.value[1], self.cfg.hyperparameters.input_shape.value[2]))
            else:
                representative_dataset = np.random.uniform(size=(200, self.cfg.hyperparameters.input_shape.value[0], self.cfg.hyperparameters.input_shape.value[1]))

            tflite_model = convert_to_tflite(model_substituted, representative_dataset)
        except Exception as e:
            raise ValueError("Error when converting to TFLite") from e
        self.my_saver.save_population_phenotype_tflite(individual_name, self.generation_counter, tflite_model)


