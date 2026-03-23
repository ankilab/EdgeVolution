"""
Shared evaluation pipeline for all search strategies.

Extracted from GeneticAlgorithm — handles translation, memory evaluation,
surrogate pre-screening, training, MCU evaluation, and fitness calculation.
"""

import logging
import os
import os.path
import json
import copy
import time
import subprocess
import numpy as np
import pandas as pd
from subprocess import Popen
from multiprocessing import Process, get_context
from tqdm import tqdm
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

from .src.translation import translate
from .src.objective_function import calculate_fitness
from .utils.convert_to_tflite import convert_to_tflite
from .utils.substitute_tflite_layer import substitute_tflite_layer
from .utils.save_ram_rom_usage import save_ram_rom_usage
from .strategies.base import Candidate, EvaluationResult

from utils.saver import Saver


class EvaluationPipeline:
    """Evaluates candidate architectures: translate, memory check, surrogate,
    train, MCU eval, fitness calculation."""

    def __init__(self, cfg: DictConfig, saver: Saver,
                 surrogate=None, hw_surrogate=None,
                 search_space_registry=None, hw_lut=None):
        self.cfg = cfg
        self.my_saver = saver
        self.surrogate = surrogate
        self.hw_surrogate = hw_surrogate
        self.search_space_registry = search_space_registry
        self.hw_lut = hw_lut

    def evaluate(self, candidates, generation,
                 use_mcu=False):
        """Run the full evaluation pipeline on a list of Candidates.

        Returns a list of EvaluationResult objects.
        """
        # Convert Candidate list to internal dict format
        individuals = {}
        for c in candidates:
            individuals[c.name] = {"genotype": c.chromosome}

        # 1. Prepare generation: translate chromosomes to models
        self._prepare_generation(individuals, generation)

        # 2. Evaluate memory footprint
        self._evaluate_memory_footprint(individuals, generation)

        # 3. Accuracy surrogate pre-screening (who skips TRAINING)
        skipped = {}
        individuals, skipped = self._surrogate_prescreen(individuals, generation)

        # 4. Hardware LUT or hardware surrogate pre-screening
        hw_skipped = {}
        all_individuals = dict(list(individuals.items()) + list(skipped.items()))

        if self.hw_lut is not None:
            # Always write LUT predictions
            self._apply_hardware_lut(all_individuals, generation)
            if use_mcu:
                # Evaluation mode: LUT predictions stored separately,
                # real MCU eval still runs and overwrites the main keys
                use_mcu_eval = True
            else:
                # Pure LUT mode: no hardware, LUT predictions used for fitness
                use_mcu_eval = False
        else:
            use_mcu_eval = use_mcu
            if use_mcu_eval:
                _, hw_skipped = self._hardware_surrogate_prescreen(
                    all_individuals, generation,
                )

        # 5. MCU evaluation + training in parallel
        mcu_individuals = {
            name: data for name, data in all_individuals.items()
            if name not in hw_skipped
        }
        if use_mcu_eval and mcu_individuals:
            process = Process(
                target=self._evaluate_mcu,
                args=(mcu_individuals, generation),
            )
            process.start()

        # Train the non-accuracy-skipped individuals
        self._train_neural_networks(individuals, generation)

        if use_mcu_eval and mcu_individuals:
            process.join()

        # 6. Merge skipped individuals back and collect surrogate data
        individuals.update(skipped)
        self._collect_surrogate_data(individuals, skipped, generation)
        self._collect_hardware_surrogate_data(individuals, hw_skipped, generation)

        # 7. Build EvaluationResult list
        return self._build_results(individuals, generation)

    # ------------------------------------------------------------------
    # Prepare generation
    # ------------------------------------------------------------------

    def _prepare_generation(self, individuals, generation):
        """Create generation directory, save genotypes, translate to models."""
        self.my_saver.create_generation_dir(individuals, generation)
        self.my_saver.save_population_genotype(individuals, generation)

        cpus = max(1, os.cpu_count() - 4)

        # Extract only the config values needed by workers to avoid pickling
        # the entire EvaluationPipeline (which includes the surrogate model).
        worker_cfg = {
            'input_shape': list(self.cfg.hyperparameters.input_shape.value),
            'num_classes': self.cfg.hyperparameters.num_classes.value,
            'top_activation': self.cfg.hyperparameters.top_activation.value,
            'sample_rate': self.cfg.hyperparameters.sample_rate.value,
            'results_dir': str(self.my_saver.results_dir),
        }

        work_items = []
        for name in individuals:
            work_items.append((name, individuals[name], generation, worker_cfg))

        with get_context("spawn").Pool(cpus) as pool:
            pool.starmap(_process_model_translation_and_conversion, work_items)

        logger.info("Finished translating and converting models")

    # ------------------------------------------------------------------
    # Memory footprint
    # ------------------------------------------------------------------

    def _evaluate_memory_footprint(self, individuals, generation):
        """Load memory footprint of converted models and write to results.json."""
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'

        individuals_copy = copy.deepcopy(individuals)
        for individual in individuals_copy.keys():
            model_path = path + individual + '/models/model_untrained.h5'
            memory_footprint_h5 = os.path.getsize(model_path)

            tflite_model_path = path + individual + '/models/model_tflite_untrained.tflite'
            memory_footprint_tflite = os.path.getsize(tflite_model_path)

            c_array_path = path + individual + '/models/model_c_array_untrained.cc'
            memory_footprint_c_array = os.path.getsize(c_array_path)

            d = {
                'memory_footprint_h5': memory_footprint_h5,
                'memory_footprint_tflite': memory_footprint_tflite,
                'memory_footprint_c_array': memory_footprint_c_array,
            }

            os.remove(c_array_path)

            with open(path + individual + '/results.json', 'w') as f:
                json.dump(d, f, indent=2)

        if len(individuals) == 0:
            raise Exception(
                "All models are too big. None will be further evaluated. "
                "Adjust your GA parameters."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_neural_networks(self, individuals, generation, max_retries=2):
        """Train all preselected models, parallelizing across GPUs.

        Failed processes (e.g. GPU OOM) are retried up to *max_retries* times.
        Retries are pushed to the back of the queue so other processes can
        finish and free GPU memory first.

        Each subprocess is killed if it exceeds ``per_process_timeout`` seconds
        (default: 1 hour).  This prevents a single hung TensorFlow process from
        blocking the entire search indefinitely.
        """
        individuals_names = list(individuals.keys())
        if not individuals_names:
            return

        tqdm_bar = tqdm(total=len(individuals_names), desc="Training models")

        # Work queue: list of names still to be trained.
        work_queue = list(individuals_names)
        # How many times each individual has been attempted.
        attempt_count = {name: 0 for name in individuals_names}
        # Track (process, individual_name, start_time) for running processes.
        active_procs = []
        permanently_failed = []
        completed_count = 0

        # Per-process timeout: kill any training subprocess that runs longer
        # than this.  Configurable via cfg.hyperparameters.training_timeout_s
        # (defaults to 3600 = 1 hour).
        per_process_timeout = 3600
        try:
            per_process_timeout = int(
                self.cfg.hyperparameters.training_timeout_s.value
            )
        except Exception:
            pass

        log_files = {}  # name -> open file handle for training logs
        gpu_available = False
        nvidia_smi = None
        handle = None
        try:
            import nvidia_smi as _nvidia_smi
            nvidia_smi = _nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total_memory = info.total
            total_gpu_gb = total_memory / (1024 ** 3)
            min_free_threshold = max(int(total_memory * 0.15), 500_000_000)
            cpu_parallel = max(1, os.cpu_count() - 2)
            # Reserve 8 GB for OS/main process, ~5 GB per training subprocess
            total_ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
            ram_parallel = max(1, int((total_ram_gb - 8) / 5))
            max_parallel = min(cpu_parallel, ram_parallel)
            constraint = "RAM" if ram_parallel < cpu_parallel else "CPU"
            gpu_available = True
            logger.info(
                f"GPU detected: {total_gpu_gb:.1f} GB total, "
                f"free-memory threshold: {min_free_threshold / 1e6:.0f} MB, "
                f"max parallel processes: {max_parallel} "
                f"({constraint}-limited: {total_ram_gb:.0f} GB RAM, {os.cpu_count()} CPUs)"
            )
        except Exception:
            max_parallel = 1
            logger.info(
                "No GPU detected or nvidia_smi unavailable. "
                "Training sequentially (1 process at a time)."
            )

        try:
            while work_queue or active_procs:
                # Reap finished / timed-out processes.
                still_alive = []
                now = time.time()
                for proc, name, start_time in active_procs:
                    retcode = proc.poll()
                    elapsed = now - start_time

                    if retcode is None and elapsed > per_process_timeout:
                        # Subprocess exceeded the timeout — kill it.
                        logger.warning(
                            f"Training of {name} timed out after "
                            f"{elapsed:.0f}s — terminating"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                        if name in log_files:
                            log_files[name].close()
                            del log_files[name]
                        self._record_training_failure(
                            name, generation, None, reason="timeout",
                        )
                        permanently_failed.append(name)
                        completed_count += 1
                        tqdm_bar.update(1)
                    elif retcode is None:
                        still_alive.append((proc, name, start_time))
                    elif retcode != 0:
                        # Close log file so we can read the full output
                        if name in log_files:
                            log_files[name].close()
                            del log_files[name]
                        if attempt_count[name] < max_retries:
                            logger.warning(
                                f"Training of {name} failed "
                                f"(exit code {retcode}), "
                                f"retrying ({attempt_count[name]}/{max_retries})"
                            )
                            work_queue.append(name)
                        else:
                            logger.warning(
                                f"Training of {name} failed "
                                f"after {max_retries} retries — giving up"
                            )
                            self._record_training_failure(
                                name, generation, retcode,
                            )
                            permanently_failed.append(name)
                            completed_count += 1
                            tqdm_bar.update(1)
                    else:
                        # Success — close log file
                        if name in log_files:
                            log_files[name].close()
                            del log_files[name]
                        completed_count += 1
                        tqdm_bar.update(1)
                active_procs = still_alive

                # Nothing left to launch — just wait for active ones.
                if not work_queue:
                    if active_procs:
                        time.sleep(10)
                    continue

                # Decide whether we can launch a new process.
                can_launch = len(active_procs) < max_parallel
                if can_launch and gpu_available:
                    try:
                        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                        if info.free <= min_free_threshold:
                            can_launch = False
                    except Exception:
                        # GPU query failed — wait for running processes to free memory.
                        can_launch = False

                if can_launch:
                    name = work_queue.pop(0)
                    attempt_count[name] += 1
                    command = [
                        'python3', 'neural_architecture_search/src/train.py',
                        '--results_dir', str(self.my_saver.results_dir),
                        '--gen_dir', f'Generation_{generation}',
                        '--individual_dir', name,
                        '--dataset', str(self.cfg.hyperparameters.dataset_name.value),
                        '--num_epochs', str(self.cfg.hyperparameters.num_epochs.value),
                        '--batch_size', str(self.cfg.hyperparameters.batch_size.value),
                        '--loss', str(self.cfg.hyperparameters.loss.value),
                        '--metrics', *[str(i) for i in self.cfg.hyperparameters.metrics.value],
                        '--optimizer', str(self.cfg.hyperparameters.optimizer.value),
                    ]
                    env = {**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3', 'TF_CUDNN_USE_AUTOTUNE': '0'}

                    is_retry = attempt_count[name] > 1
                    if is_retry:
                        logger.info(f"Retrying training of {name} "
                                    f"(attempt {attempt_count[name]}/{max_retries + 1})")

                    # Capture stdout/stderr to a log file for debugging
                    individual_dir = f'{self.my_saver.results_dir}/Generation_{generation}/{name}'
                    log_path = os.path.join(individual_dir, 'training_log.txt')
                    log_file = open(log_path, 'a')
                    log_files[name] = log_file

                    proc = Popen(command, env=env, stdout=log_file, stderr=log_file)
                    active_procs.append((proc, name, time.time()))
                    # Pause to let TF allocate GPU memory before checking free space.
                    time.sleep(2)
                else:
                    # GPU busy or at max parallel — wait before retrying.
                    time.sleep(3)

            if permanently_failed:
                logger.warning(f"{len(permanently_failed)}/{len(individuals_names)} "
                               f"training processes permanently failed: {permanently_failed}")
        finally:
            # Kill all still-running training subprocesses.
            for proc, name, _start in active_procs:
                if proc.poll() is None:
                    logger.info(f"Terminating training of {name}...")
                    proc.terminate()
            for proc, name, _start in active_procs:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            # Close any remaining log file handles
            for name, lf in log_files.items():
                try:
                    lf.close()
                except Exception:
                    pass

            if gpu_available and nvidia_smi is not None:
                try:
                    nvidia_smi.nvmlShutdown()
                except Exception:
                    pass

    def _record_training_failure(self, name, generation, exit_code,
                                  reason="training_failed"):
        """Read training log and write error details to results.json."""
        individual_dir = f'{self.my_saver.results_dir}/Generation_{generation}/{name}'
        log_path = os.path.join(individual_dir, 'training_log.txt')

        # Read last lines from training log for context
        error_tail = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    # Keep last 30 lines for error context
                    error_tail = "".join(lines[-30:]).strip()
            except Exception:
                error_tail = "<could not read training log>"

        # Update results.json with error info
        results_path = os.path.join(individual_dir, 'results.json')
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except Exception:
            results = {}

        # Merge with any error info already written by train.py
        existing_error = results.get('training_error', {})
        if not isinstance(existing_error, dict):
            existing_error = {}
        existing_error.update({
            'reason': reason,
            'exit_code': exit_code,
            'log_tail': error_tail,
        })
        results['training_error'] = existing_error

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        if error_tail:
            logger.warning(f"Training error for {name} (exit={exit_code}): "
                           f"...{error_tail[-200:]}")

    # ------------------------------------------------------------------
    # Surrogate pre-screening
    # ------------------------------------------------------------------

    def _surrogate_prescreen(self, individuals, generation):
        """Use surrogate model to skip low-confidence individuals.

        Returns (individuals_to_train, skipped_individuals).
        """
        if self.surrogate is None or self.search_space_registry is None:
            return individuals, {}

        to_train, to_skip, _ = self.surrogate.prescreen(
            individuals, self.search_space_registry.encode
        )

        if not to_skip:
            return individuals, {}

        predictions = self.surrogate.get_predictions()
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'

        skipped = {}
        for name in to_skip:
            skipped[name] = individuals[name]
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

        remaining = {
            name: data for name, data in individuals.items()
            if name in to_train
        }
        return remaining, skipped

    def _collect_surrogate_data(self, individuals, skipped, generation):
        """Collect actual accuracies, update surrogate, log generation data."""
        if self.surrogate is None or self.search_space_registry is None:
            return

        path = f'{self.my_saver.results_dir}/Generation_{generation}/'
        predictions = self.surrogate.get_predictions()

        # Collect observations from trained individuals (not skipped)
        trained_names = set(individuals.keys()) - set(skipped.keys())
        for name in trained_names:
            data = individuals[name]
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
        for name in individuals.keys():
            pred = predictions.get(name, {})
            predicted_acc = pred.get("predicted_acc")
            uncertainty = pred.get("uncertainty")
            is_skipped = name in skipped

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
                "skipped": is_skipped,
            })

        surrogate_dir = f'{self.my_saver.results_dir}/surrogate_accuracy'
        self.surrogate.log_generation(generation, individual_records, surrogate_dir)

        if self.surrogate.is_ready:
            self.surrogate.fit()
            self.surrogate.save(surrogate_dir)

    # ------------------------------------------------------------------
    # Hardware surrogate pre-screening
    # ------------------------------------------------------------------

    def _hardware_surrogate_prescreen(self, all_individuals, generation):
        """Use hardware surrogate to skip MCU evaluation for some individuals.

        For skipped individuals, writes predicted energy and tflite-based ROM
        proxy to results.json.

        Args:
            all_individuals: All individuals (both training and accuracy-skipped).
            generation: Current generation number.

        Returns:
            (remaining, hw_skipped) — remaining need real MCU eval,
            hw_skipped have predicted energy written to results.json.
        """
        if self.hw_surrogate is None or self.search_space_registry is None:
            return all_individuals, {}

        to_eval, to_skip, _ = self.hw_surrogate.prescreen(
            all_individuals, self.search_space_registry.encode
        )

        if not to_skip:
            return all_individuals, {}

        predictions = self.hw_surrogate.get_predictions()
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'

        # Get board SNR for writing energy_information dict
        snr = self.cfg.boards.value[0].snr if len(self.cfg.boards.value) > 0 else "unknown"

        hw_skipped = {}
        for name in to_skip:
            hw_skipped[name] = all_individuals[name]
            pred = predictions.get(name, {})
            predicted_energy = pred.get("predicted_acc", 0.0)  # generic predicted value

            results_path = path + name + '/results.json'
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                results = {}

            # Use tflite size as ROM proxy when MCU eval is skipped
            results['rom_usage'] = results.get('memory_footprint_tflite', 0)
            results['energy_information'] = {snr: predicted_energy}
            results['hw_surrogate_skipped'] = True

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        remaining = {
            name: data for name, data in all_individuals.items()
            if name in to_eval
        }
        return remaining, hw_skipped

    def _collect_hardware_surrogate_data(self, individuals, hw_skipped, generation):
        """Collect real energy data from MCU-evaluated individuals, update
        hardware surrogate, and log generation data."""
        if self.hw_surrogate is None or self.search_space_registry is None:
            return

        path = f'{self.my_saver.results_dir}/Generation_{generation}/'
        predictions = self.hw_surrogate.get_predictions()

        snr = self.cfg.boards.value[0].snr if len(self.cfg.boards.value) > 0 else None

        # Collect observations from MCU-evaluated individuals (not hw-skipped)
        evaluated_names = set(individuals.keys()) - set(hw_skipped.keys())
        for name in evaluated_names:
            data = individuals[name]
            results_path = path + name + '/results.json'
            if not os.path.exists(results_path):
                continue
            with open(results_path, 'r') as f:
                results = json.load(f)

            energy_info = results.get('energy_information')
            if isinstance(energy_info, dict) and snr is not None:
                energy_val = energy_info.get(snr)
                if isinstance(energy_val, (int, float)):
                    encoding = self.search_space_registry.encode(data["genotype"])
                    self.hw_surrogate.add_observation(encoding, float(energy_val))

        # Build per-individual records for logging
        individual_records = []
        for name in individuals.keys():
            pred = predictions.get(name, {})
            predicted_val = pred.get("predicted_acc")
            uncertainty = pred.get("uncertainty")
            is_skipped = name in hw_skipped

            results_path = path + name + '/results.json'
            actual_val = None
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                energy_info = results.get('energy_information')
                if isinstance(energy_info, dict) and snr is not None:
                    val = energy_info.get(snr)
                    if isinstance(val, (int, float)):
                        actual_val = float(val)

            individual_records.append({
                "name": name,
                "predicted_acc": predicted_val,
                "uncertainty": uncertainty,
                "actual_acc": actual_val,
                "skipped": is_skipped,
            })

        surrogate_hw_dir = f'{self.my_saver.results_dir}/surrogate_hardware'
        self.hw_surrogate.log_generation(generation, individual_records, surrogate_hw_dir)

        if self.hw_surrogate.is_ready:
            self.hw_surrogate.fit()
            self.hw_surrogate.save(surrogate_hw_dir)

    # ------------------------------------------------------------------
    # MCU evaluation
    # ------------------------------------------------------------------

    def _evaluate_mcu(self, individuals, generation):
        """Evaluate models on MCU: flash, measure inference time and energy."""
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'

        from tools.measure_power_consumption import init_ppk2, stop_measuring

        individuals_names = list(individuals.keys())
        for idx, individual in tqdm(enumerate(individuals_names), total=len(individuals_names)):
            logger.info(f"Evaluate energy of {individual} (index: {idx + 1})")

            error_log_path = path + individual + '/error_log.txt'

            if len(self.cfg.boards.value) > 0:
                for board in self.cfg.boards.value:
                    # Use absolute paths so they resolve correctly after cd in flash script
                    tflite_path = os.path.abspath(path + individual + '/models/model_tflite_untrained.tflite')
                    cpp_path = os.path.abspath('tflite/edgevolution_tflite/src/model.cpp')
                    flasher_path = os.path.abspath('tools/flash_tflite_model.sh')
                    results_path = path + individual + '/results.json'

                    ppk2 = init_ppk2(board.ppk)
                    time.sleep(2)

                    ret_val = -1
                    try:
                        ret_val = subprocess.call(
                            ['bash', flasher_path, tflite_path, cpp_path, board.model, board.snr]
                        )
                    except Exception as e:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error flashing model on board {board.snr} - exception: {str(e)}.\n")

                    if ret_val != 0:
                        with open(error_log_path, 'a') as f:
                            f.write(f"Error flashing model on board {board.snr}. Ret val: {ret_val}.\n")
                        # Write flash error to results.json so calculate_fitness sees it
                        try:
                            with open(results_path) as f:
                                results = json.loads(f.read())
                            results['flash_error'] = f"Flash failed on board {board.snr} (ret_val={ret_val})"
                            with open(results_path, 'w') as f:
                                json.dump(results, f, indent=2)
                        except Exception:
                            pass
                        del ppk2
                        time.sleep(3)
                        continue

                    time.sleep(5)

                    save_ram_rom_usage(
                        "tflite/build-" + board.model,
                        path + individual + "/" + "results.json",
                    )

                    proc_energy = None
                    if ppk2 is not None:
                        del ppk2
                        time.sleep(3)

                        args = [
                            'python3 tools/measure_power_consumption.py',
                            path + individual, board.snr, board.ppk,
                            f'{self.cfg.hyperparameters.power_measurement_num_samples_average.value}',
                        ]
                        command = " ".join(args)
                        proc_energy = Popen(command, shell=True)
                        time.sleep(2)

                    try:
                        args = [
                            'python3 tools/measure_inference_time.py',
                            path + individual, board.model, board.snr,
                        ]
                        command = " ".join(args)
                        proc_inference = Popen(command, shell=True)
                        proc_inference.wait()
                    except Exception as e:
                        with open(error_log_path, 'a') as f:
                            f.write(
                                f"Error measuring inference time on board {board.snr}.\n"
                                f" Exception: {str(e)}\n"
                            )

                    if proc_energy is not None:
                        try:
                            proc_energy.wait(timeout=10)
                        except Exception:
                            with open(error_log_path, 'a') as f:
                                f.write(f"Error measuring energy on board {board.snr}.\n")

                        try:
                            self._calculate_energy_consumption(
                                board.snr, board.power_measurement_threshold,
                                path + individual,
                            )
                        except Exception as e:
                            with open(error_log_path, 'a') as f:
                                f.write(
                                    f"Error calculating energy on board {board.snr}.\n"
                                    f"Exception: {str(e)}\n"
                                )
                    time.sleep(3)
            else:
                raise ValueError(
                    f'No boards set. Length of params["boards"]: {len(self.cfg.boards.value)}'
                )

    def _calculate_energy_consumption(self, board_snr, power_measurement_threshold, data_dir):
        """Calculate energy consumption from power measurement CSV."""
        power_measurement_file_name = "power_measurements_" + board_snr + ".csv"
        csv_path = os.path.join(data_dir, power_measurement_file_name)
        results_path = os.path.join(data_dir, "results.json")

        try:
            with open(results_path) as f:
                results = json.loads(f.read())
        except FileNotFoundError:
            raise NotImplementedError(
                "Not implemented proper handling if result does not exist."
            )
        except Exception:
            raise NotImplementedError("proper error handling")

        try:
            data = pd.read_csv(csv_path)
            values = np.asarray(data["Power Consumption"])
            values = values[10000:]

            values_averaged = (
                pd.Series(values)
                .rolling(self.cfg.hyperparameters.power_measurement_num_samples_average.value)
                .mean()
            )

            # Auto-detect inference window from power trace.
            # Try to find idle→active transition; if recording started
            # during active inference (no idle region), use the full trace.
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
                # No clear transition: MCU was already active during recording.
                # Use mean of full trace (excluding NaN from rolling window).
                start = 0
                end = len(values_averaged)

            valid = values[start:end]
            valid = valid[~np.isnan(valid)]
            mean_power_consumption = np.mean(valid) if len(valid) > 0 else np.nanmean(values)  # uA
            mean_power_consumption = mean_power_consumption * (10 ** -6)  # A

            voltage = 3.3  # V
            inf_time = _get_inference_information_from_results(board_snr, results)  # ms
            inf_time = inf_time * (10 ** -3)  # s

            energy_consumption = voltage * mean_power_consumption * inf_time  # J
            energy_consumption = energy_consumption * (10 ** 3)  # mJ

            results["energy_information"] = _set_result_value_for_board(
                board_snr, "energy_information", float(energy_consumption), results
            )
            results["mean_power_information"] = _set_result_value_for_board(
                board_snr, "mean_power_information", float(mean_power_consumption), results
            )
        except Exception as e:
            results["energy_information"] = _set_result_value_for_board(
                board_snr, "energy", str(e), results
            )

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Hardware LUT
    # ------------------------------------------------------------------

    def _apply_hardware_lut(self, individuals, generation):
        """Use a pre-built hardware LUT to predict MCU metrics.

        When real MCU eval will follow (use_mcu=True), predictions are
        stored under ``hw_lut_*`` keys for comparison.  Otherwise they
        are written directly to the main metric keys and used for fitness.
        """
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'
        snr = self.cfg.boards.value[0].snr if len(self.cfg.boards.value) > 0 else "unknown"
        use_mcu = self.cfg.hyperparameters.optimize_for_MCU.value

        for name, data in individuals.items():
            chromosome = data["genotype"]
            encoding = None
            if self.search_space_registry is not None:
                encoding = self.search_space_registry.encode(chromosome)

            predictions = self.hw_lut.predict(chromosome, encoding)

            results_path = path + name + '/results.json'
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                results = {}

            if use_mcu:
                # Evaluation mode: store predictions separately,
                # real MCU eval will write the main keys
                if 'energy' in predictions:
                    results['hw_lut_energy'] = predictions['energy'][0]
                if 'inference_time' in predictions:
                    results['hw_lut_inference_time'] = predictions['inference_time'][0]
                if 'rom' in predictions:
                    results['hw_lut_rom'] = int(predictions['rom'][0])
            else:
                # Pure LUT mode: write directly to main keys
                if 'energy' in predictions:
                    results['energy_information'] = {snr: predictions['energy'][0]}
                if 'inference_time' in predictions:
                    results['inference_information'] = {snr: predictions['inference_time'][0]}
                if 'rom' in predictions:
                    results['rom_usage'] = int(predictions['rom'][0])

            results['hw_lut_predicted'] = True

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        logger.info(f"Hardware LUT applied to {len(individuals)} individuals"
                    f"{' (evaluation mode)' if use_mcu else ''}.")

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------

    def _build_results(self, individuals, generation):
        """Calculate fitness and build EvaluationResult list."""
        path = f'{self.my_saver.results_dir}/Generation_{generation}/'
        results_list = []

        for name, data in individuals.items():
            results_path = path + name + '/results.json'
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
            except Exception:
                results = {}

            fitness, error = calculate_fitness(results, self.cfg)

            # Write fitness back to results.json
            results['fitness'] = float(fitness)
            results['error'] = str(error)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            results_list.append(EvaluationResult(
                name=name,
                chromosome=data["genotype"],
                val_acc=results.get('val_acc', 0.0),
                memory_footprint_tflite=results.get('memory_footprint_tflite', 0),
                fitness=fitness,
                energy_consumption=self._extract_energy(results),
                inference_time=self._extract_inference_time(results),
                rom_usage=results.get('rom_usage'),
                surrogate_predicted=results.get('surrogate_skipped', False),
                hw_surrogate_predicted=results.get('hw_surrogate_skipped', False),
                hw_lut_predicted=results.get('hw_lut_predicted', False),
                error=error,
            ))

        return results_list

    def _extract_energy(self, results):
        """Extract energy consumption from results dict."""
        energy_info = results.get('energy_information')
        if isinstance(energy_info, dict):
            values = [v for v in energy_info.values() if isinstance(v, (int, float))]
            return values[0] if values else None
        return None

    def _extract_inference_time(self, results):
        """Extract inference time from results dict."""
        inf_info = results.get('inference_information')
        if isinstance(inf_info, dict):
            values = [v for v in inf_info.values() if isinstance(v, (int, float))]
            return values[0] if values else None
        return None



# ------------------------------------------------------------------
# Pool worker (module-level to avoid pickling the entire pipeline)
# ------------------------------------------------------------------

def _process_model_translation_and_conversion(individual_name, individual_data, generation, worker_cfg):
    """Translate a chromosome to a TensorFlow model and convert to TFLite.

    This is a module-level function (not a bound method) so that
    multiprocessing only pickles the lightweight worker_cfg dict
    instead of the entire EvaluationPipeline instance.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    input_shape = worker_cfg['input_shape']
    results_dir = worker_cfg['results_dir']

    try:
        model = translate(
            individual_data['genotype'],
            input_shape,
            worker_cfg['num_classes'],
            worker_cfg['top_activation'],
            worker_cfg['sample_rate'],
        )
    except Exception:
        raise ValueError(
            f"Error translating genotype to phenotype. "
            f"Chromosome: {individual_data['genotype']}"
        )

    # Save .h5 model
    from pathlib import Path
    model_dir = Path(results_dir) / f'Generation_{generation}' / individual_name / 'models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir / 'model_untrained.h5')

    try:
        model_substituted = substitute_tflite_layer(model, input_shape)
    except Exception:
        raise ValueError("Error substituting STFT and MAG layers.")

    try:
        if len(input_shape) == 3:
            representative_dataset = np.random.uniform(
                size=(200, input_shape[0], input_shape[1], input_shape[2])
            )
        else:
            representative_dataset = np.random.uniform(
                size=(200, input_shape[0], input_shape[1])
            )
        tflite_model = convert_to_tflite(model_substituted, representative_dataset)
    except Exception:
        raise ValueError("Error converting to TFLite")

    logger.debug(f"Save TFLite model of {individual_name}")

    # Save .tflite model
    tflite_path = model_dir / 'model_tflite_untrained.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Convert tflite model to C-array
    try:
        subprocess.call(
            "xxd -i " + str(tflite_path) + " > " + str(model_dir / "model_c_array_untrained.cc"),
            shell=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        logger.warning("xxd command timed out")


# ------------------------------------------------------------------
# Static helpers (moved from GeneticAlgorithm)
# ------------------------------------------------------------------

def _get_inference_information_from_results(board_snr, results):
    """Read inference time for a board from results dict."""
    if "inference_information" not in results:
        raise ValueError("key 'inference_information' does not exist")
    if board_snr not in results["inference_information"]:
        raise ValueError("board_snr does not exist in inference information")
    return float(results["inference_information"][board_snr])


def _set_result_value_for_board(board_snr, category, value, results):
    """Set a board-specific result value."""
    _id = board_snr
    information = {}
    if category in results:
        information = results[category]
        if _id in information:
            raise RuntimeError(
                f"results.json already contains {category} of board {board_snr}"
            )
    information[_id] = value
    return information
