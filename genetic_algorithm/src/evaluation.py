import numpy as np


def get_fitness(weighting_factors, objectives):
    return np.sum(weighting_factors * objectives)