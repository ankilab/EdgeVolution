
def calculate_fitness(results, params):
    a = 0.7  # weight test_acc
    try:
        fitness = a * results['val_acc']
        return fitness
    except:
        # if a key does not exist, the model was not evaluated correctly --> there was something wrong with the model,
        # so we omit it for crossover following generations through giving it a bad fitness
        return -10001
