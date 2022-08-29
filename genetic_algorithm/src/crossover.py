import numpy as np
import json


def crossover(path, fittest_chromosomes, population_size):
    new_population = []
    while len(new_population) < population_size:
        # load gene information of two random chromosomes
        chromosome_1_name = np.random.choice(fittest_chromosomes)
        chromosome_2_name = np.random.choice([gene for gene in fittest_chromosomes if gene != chromosome_1_name])
        with open(path + chromosome_1_name + '/chromosome.json') as f:
            chromosome_1 = json.loads(f.read())
        with open(path + chromosome_2_name + '/chromosome.json') as f:
            chromosome_2 = json.loads(f.read())

        #
        chr_1_idx = get_flatten_gap_gmp_index(chromosome_1)
        chr_2_idx = get_flatten_gap_gmp_index(chromosome_2)

        #chr_1_split_1, chr_1_split_2 = rnd(chromosome_1[0:chr_1_idx]), rnd(chromosome_1[chr_1_idx::])
        #chr_2_split_1, chr_2_split_2 = rnd(chromosome_2[0:chr_2_idx]), rnd(chromosome_2[chr_2_idx::])

        # new_chromosome_1 = chromosome_1[0:chr_1_split_1] + chromosome_2[chr_2_split_1:chr_2_idx] \
        #                    + chromosome_1[chr_1_idx:chr_1_split_2] + chromosome_2[chr_2_split_2::]

        new_chromosome_1 = chromosome_1[0:chr_1_idx] + chromosome_2[chr_2_idx::]
        new_chromosome_2 = chromosome_2[0:chr_2_idx] + chromosome_1[chr_1_idx::]
        new_population.append(new_chromosome_1)
        new_population.append(new_chromosome_2)
    return new_population


def get_flatten_gap_gmp_index(chromosome):
    for idx, gene in enumerate(chromosome):
        if gene['layer'] == 'F' or gene['layer'] == 'GAP' or gene['layer'] == 'GMP':
            return idx


def rnd(_list):
    return np.random.randint(1, len(_list))
