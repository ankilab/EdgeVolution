

def calculate_fitness(results, params):
    a = 0.4  # weight test_acc
    b = 0.2  # weight memory_footprint_tflite
    c = 0.2  # weight inference_time
    d = 0.2  # weight energy_consumption
    try:
        fitness = a * results['val_acc'] + \
                  b * (1 - (results["memory_footprint_tflite"] / params["max_memory_footprint"])) + \
                  c * (1 - (results["inference_time"] / params["max_inference_time"])) + \
                  d * (1 - (results["energy_consumption"] / params["max_energy_consumption"]))
        return fitness
    except:
        # if key does not exist, the model was not trained --> there was something wrong with the model, so we omit it
        # for crossover following generations through giving it a bad fitness
        return -12345
