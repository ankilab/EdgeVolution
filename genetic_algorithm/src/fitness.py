from omegaconf import DictConfig

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

    # Check if fitness only consists of accuracy (i.e. no MCU evaluation)
    if cfg.boards.value[0].model is None:
        return results['val_acc'], False

    snr = cfg.boards.value[0].snr
    try:
        acc = results['val_acc']
        rom_usage = results["rom_usage"]
        energy_information = results["energy_information"][snr]

        acc_weighted = acc * cfg.hyperparameters.acc_weight.value

        rom_usage_scaled = (cfg.hyperparameters.min_rom_usage.value / rom_usage)
        if rom_usage_scaled > 1:
            rom_usage_scaled = 1
        rom_usage_weighted = rom_usage_scaled * cfg.hyperparameters.rom_usage_weight.value
        
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
        print(f'caught {type(e)}: error in calculate_fitness')
        error = True
        return 0, error
