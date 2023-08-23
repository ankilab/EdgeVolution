from genetic_algorithm.src.fitness import calculate_fitness


def test_calculate_fitness():
    results = {}
    results["val_acc"] = 0.95
    results["memory_footprint_tflite"] = 800_000
    results["inference_time"] = 200
    results["energy_consumption"] = 2
    params ={}
    params["max_memory_footprint"] = 800_000
    params["max_inference_time"] = 200
    params["max_energy_consumption"] = 2
    fitness = calculate_fitness(results,params)
    assert(fitness == 0.7 * 0.95)

def test_calculate_fitness_empty():
    results = {}
    results["val_acc"] = 0.95
    results["memory_footprint_tflite"] = 100_000
    results["inference_time"] = 122
    results["energy_consumption"] = 1
    params ={}
    
    # missing on purpose
    #params["max_memory_footprint"] = 800_000

    params["max_inference_time"] = 200
    params["max_energy_consumption"] = 2
    calculate_fitness(results,params)
    fitness = calculate_fitness(results,params)
    assert(fitness == -10001)
