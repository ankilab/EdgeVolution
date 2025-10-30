import tensorflow as tf
import multiprocessing
import numpy as np
import hydra
from omegaconf import DictConfig

from neural_architecture_search.genetic_algorithm import GeneticAlgorithm
from neural_architecture_search.utils.profiling import ProfilingStats, time_phase, calculate_summary_statistics
from utils.saver import Saver
from tools.update_tensor_arena_size import update_tensor_arena_size


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Limit TensorFlow GPU memory usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_saver = Saver(cfg.hyperparameters.results_path.value, cfg.hyperparameters.dataset_name.value)

    # Initialize profiling with save directory for incremental saves
    profiling_stats = ProfilingStats(save_dir=str(my_saver.results_dir), verbose=True)
    profiling_stats.start_overall_run()

    my_ga = GeneticAlgorithm(cfg, my_saver, profiling_stats=profiling_stats)

    # random init the population of the first generation
    my_ga.init_first_generation()
    gen_start = 1

    # save params
    my_saver.save_params(cfg)
    
    # Check if MCU evaluation is enabled
    mcu_evaluation_enabled = cfg.boards.value[0].model is not None
    
    if mcu_evaluation_enabled:
        # update the tensor arena size in the main.cpp file
        limit_tensor_arena_size = _get_tensor_arena_size_limit(cfg)
        update_tensor_arena_size("tflite/edgevolution_tflite/src/main_functions.cpp", limit_tensor_arena_size)
        print(f"MCU evaluation enabled - using board: {cfg.boards.value[0].model}")
    else:
        print("MCU evaluation disabled - running accuracy-only NAS")

    for i_generation in range(gen_start, cfg.hyperparameters.num_generations.value + 1):
        print(f"\n{'='*80}")
        print(f"Starting Generation {i_generation}/{cfg.hyperparameters.num_generations.value}")
        print(f"{'='*80}")
        
        profiling_stats.start_generation(i_generation)
        
        with time_phase(profiling_stats, i_generation, "prepare_generation"):
            my_ga.prepare_generation(i_generation)

        # Pre-selection of candidate chromosomes, which are trained on a GPU afterwards
        with time_phase(profiling_stats, i_generation, "evaluate_memory_footprint"):
            my_ga.evaluate_memory_footprint()

        if mcu_evaluation_enabled:
            # Evaluate candidate models on MCU (i.e. flash them to MCU and measure objectives)
            # this will start a process that is constantly running and evaluating an individual after training is finished
            process = multiprocessing.Process(target=my_ga.evaluate_energy_consumption_and_inference_speed)
            process.start()

        # train all neural networks 
        with time_phase(profiling_stats, i_generation, "train_neural_networks"):
            my_ga.train_neural_networks()

        if mcu_evaluation_enabled:
            # wait for the process to finish
            with time_phase(profiling_stats, i_generation, "mcu_evaluation_wait"):
                process.join()
            
            # Load deployment times from the MCU evaluation process and add to profiling
            deployment_times_file = my_saver.results_dir / 'deployment_times.json'
            if deployment_times_file.exists():
                try:
                    import json
                    with open(deployment_times_file, 'r') as f:
                        deployment_data = json.load(f)
                    
                    if deployment_data.get('generation') == i_generation:
                        for model_name, deployment_time in deployment_data.get('deployment_times', {}).items():
                            profiling_stats.record_model_operation(
                                i_generation, model_name, "mcu_deployment", deployment_time
                            )
                except Exception as e:
                    print(f"Warning: Could not load deployment times: {e}")

        # determine the fitness for each model and select the best ones
        with time_phase(profiling_stats, i_generation, "selection"):
            my_ga.selection()

        # Preparation of the next generation, unless we have just run the last generation
        if i_generation != cfg.hyperparameters.num_generations.value:
            with time_phase(profiling_stats, i_generation, "crossover"):
                my_ga.crossover()
            with time_phase(profiling_stats, i_generation, "mutation"):
                my_ga.mutation()
        
        profiling_stats.end_generation(i_generation)
    
    # Finalize profiling and save statistics
    profiling_stats.end_overall_run()
    
    # Save detailed profiling data
    my_saver.save_profiling_stats(profiling_stats)
    
    # Calculate and save summary statistics
    summary = calculate_summary_statistics(profiling_stats)
    my_saver.save_profiling_summary(summary)
            
def _get_tensor_arena_size_limit(cfg):
    board_available_tensor_arena_size = cfg.boards.value[0].max_available_tensor_arena_size
    limit_tensor_arena_size = cfg.hyperparameters.limit_tensor_arena_size.value
    if limit_tensor_arena_size is None:
        limit_tensor_arena_size = board_available_tensor_arena_size
    else:
        limit_tensor_arena_size = min(limit_tensor_arena_size, board_available_tensor_arena_size)
    return limit_tensor_arena_size


if __name__ == "__main__":
    np.random.seed(42)
    main()
