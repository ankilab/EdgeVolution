

def calculate_fitness(results, params):
    # return (results['test_acc']
    # + (1 - results['energy_consumption'] * 1/params['max_inference_speed'])
    # + (1 - results['inference_speed'] * 1/params['max_power_consumption']))

    try:
        return results['test_acc']
    except:
        # if key does not exist, the model was not trained --> there was something wrong with the model, so we omit it
        # for crossover following generations through giving it a bad fitness
        return -1
