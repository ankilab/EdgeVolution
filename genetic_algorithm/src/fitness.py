

def calculate_fitness(results, params):
    try:
        fitness = results['test_acc'] + \
                  (1 - (results["memory_footprint_tflite"] / params["max_memory_footprint"])) + \
                  (1 - (results["inference_time"] / params["max_inference_time"])) + \
                  (1 - (results["energy_consumption"] / params["max_energy_consumption"]))
        return fitness
    except:
        # if key does not exist, the model was not trained --> there was something wrong with the model, so we omit it
        # for crossover following generations through giving it a bad fitness
        return -12345
