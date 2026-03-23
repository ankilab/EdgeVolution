import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def calculate_fitness(results, cfg: DictConfig):
    """
    Calculate the fitness of a model based on the results of the evaluation.
    The fitness is calculated based on the accuracy, the ROM usage and the energy information.
    The fitness is a weighted sum of the three values.
    The weights are defined in the config file.

    :param results: The results of the evaluation
    :param cfg: The config file
    :return: The fitness value and an error flag
    """

    # Check for training error and log it
    training_error = results.get('training_error')
    if training_error:
        error_detail = training_error.get('exception', training_error.get('reason', 'unknown'))
        logger.warning(f"calculate_fitness: training failed — {error_detail}")
        return 0, True

    # Check for flash/MCU error
    flash_error = results.get('flash_error')
    if flash_error:
        logger.warning(f"calculate_fitness: flash failed — {flash_error}")
        return 0, True

    # Check for MCU inference/energy errors (non-numeric values indicate failure)
    if cfg.boards.value and cfg.boards.value[0].model is not None:
        snr_check = cfg.boards.value[0].snr
        inf_info = results.get('inference_information', {})
        if isinstance(inf_info, dict):
            inf_val = inf_info.get(snr_check)
            if inf_val is not None and not isinstance(inf_val, (int, float)):
                logger.warning(f"calculate_fitness: MCU inference failed — {inf_val}")
                return 0, True

    # Check if fitness only consists of accuracy (i.e. no MCU evaluation)
    if cfg.boards.value[0].model is None:
        if 'val_acc' not in results:
            logger.warning(f"calculate_fitness: 'val_acc' missing from results "
                           f"(available keys: {list(results.keys())})")
            return 0, True
        return results['val_acc'], False

    snr = cfg.boards.value[0].snr
    try:
        acc = results['val_acc']

        # ROM usage: fall back to memory_footprint_tflite when rom_usage
        # is missing (e.g. when hardware surrogate skips MCU evaluation).
        rom_usage = results.get("rom_usage")
        if rom_usage is None:
            rom_usage = results.get("memory_footprint_tflite")
        if rom_usage is None or rom_usage == 0:
            logger.warning("rom_usage and memory_footprint_tflite both missing/zero — using pessimistic ROM contribution")
            rom_usage_weighted = 0
        else:
            rom_usage_scaled = (cfg.hyperparameters.min_rom_usage.value / rom_usage)
            if rom_usage_scaled > 1:
                rom_usage_scaled = 1
            rom_usage_weighted = rom_usage_scaled * cfg.hyperparameters.rom_usage_weight.value

        acc_weighted = acc * cfg.hyperparameters.acc_weight.value

        # Energy information: default to 0 contribution when missing
        # (e.g. when hardware surrogate is not active and MCU eval is skipped).
        energy_info = results.get("energy_information")
        if energy_info is None:
            energy_information = 0
        elif isinstance(energy_info, dict):
            energy_information = energy_info.get(snr, 0)
            if not isinstance(energy_information, (int, float)):
                energy_information = 0
        else:
            energy_information = 0

        if energy_information == 0:
            energy_information_scaled = 0
        else:
            energy_information_scaled = (cfg.hyperparameters.min_energy_information.value / energy_information)
            if energy_information_scaled > 1:
                energy_information_scaled = 1

        energy_information_weighted = energy_information_scaled * cfg.hyperparameters.energy_information_weight.value

        fitness = acc_weighted + rom_usage_weighted + energy_information_weighted

        error = False
        return fitness, error
    except Exception as e:
        logger.warning(f"calculate_fitness error: {type(e).__name__}: {e} "
                        f"(available keys: {list(results.keys())})")
        return 0, True
