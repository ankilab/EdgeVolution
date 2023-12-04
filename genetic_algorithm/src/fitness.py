

def calculate_fitness(results, params):
    a = 0.7  # weight val_acc
    b = 0.1  # weight rom_usage (flash memory)
    c = 0.1  # weight inference_time
    d = 0.1  # weight energy_consumption

    snr = params['boards'][0]['snr']
    try:
        fitness = a * results['val_acc'] + \
                  b * (1 - (results["rom_usage"] / params["max_rom_usage"])) + \
                  c * (1 - (results["inference_information"][snr] / params["max_inference_time"])) + \
                  d * (1 - (results["energy_information"][snr] / params["max_energy_consumption"]))
        return fitness
    except:
        # if a key does not exist, the model was not evaluated correctly --> there was something wrong with the model,
        # so we omit it for crossover following generations through giving it a bad fitness
        return -10001
