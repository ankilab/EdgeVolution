

def calculate_fitness(results, params):
    a = 0.7  # weight test_acc
    b = 0.1  # weight memory_footprint_tflite
    c = 0.1  # weight inference_time
    d = 0.1  # weight energy_consumption
    try:
        fitness = a * results['val_acc'] + \
                  b * (1 - (results["memory_footprint_tflite"] / params["max_memory_footprint"])) + \
                  c * (1 - (results["inference_information"]["1050289157"] / params["max_inference_time"])) + \
                  d * (1 - (results["energy_information"]["1050289157"] / params["max_energy_consumption"]))
        return fitness
    except:
        # if a key does not exist, the model was not evaluated correctly --> there was something wrong with the model,
        # so we omit it for crossover following generations through giving it a bad fitness
        return -10001
