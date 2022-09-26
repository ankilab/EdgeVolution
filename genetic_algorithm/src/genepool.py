from ast import literal_eval
import numpy as np
import json


class GenePool:
    def __init__(self, params):
        self.params = params
        with open(params['path_gene_pool'], "r") as f:
            self.gene_pool = literal_eval(f.read())

        with open(params['path_rule_set'], "r") as f:
            self.rule_set = literal_eval(f.read())

    def get_random_chromosome(self):
        # create a gene sequence containing all layers of this random gene
        # add preprocessing layers (STFT, normalization and Resizing)
        # chromosome = [self._get_gene_with_random_parameters('STFT'), self._get_gene_with_random_parameters('NORM'),
                      # self._get_gene_with_random_parameters('RES')]

        # first layer after preprocessing layers is always 'C' or 'DC'
        gene = np.random.choice(['C', 'DC'])
        chromosome = []
        chromosome.append(self._get_gene_with_random_parameters(gene))

        for _ in range(1, np.random.choice(np.arange(1, self.params['max_nb_feature_layers'] + 1))):
            possible_genes = self._get_possible_genes(gene)  # check rule set
            gene = np.random.choice(possible_genes)

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(gene))

        gene = np.random.choice(['GAP', 'GMP', 'F'])
        chromosome.append(self._get_gene_with_random_parameters(gene))

        for _ in range(1, np.random.choice(np.arange(1, self.params['max_nb_classification_layers'] + 1))):
            possible_genes = self._get_possible_genes(gene)  # check rule set
            gene = np.random.choice(possible_genes)

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(gene))

        return chromosome

    def _get_gene_with_random_parameters(self, target_gene: str) -> dict:
        """ Method to get random parameters for a given gene. """
        gene_with_random_params = {}
        for gene in self.gene_pool:
            # find target_gene in gene pool
            if gene['layer'] == target_gene:
                # iterate over each property of the gene and get random value for it
                for _property in gene:
                    if _property == 'layer' or _property == 'f_name':
                        gene_with_random_params[_property] = gene[_property]
                    elif type(gene[_property][0]) is int:
                        gene_with_random_params[_property] = int(np.random.choice(np.arange(gene[_property][0],
                                                                                            gene[_property][1] + 1,
                                                                                            gene[_property][2])))
                    elif type(gene[_property][0]) is float:
                        gene_with_random_params[_property] = float(np.random.choice(np.arange(gene[_property][0],
                                                                                              gene[_property][1],
                                                                                              gene[_property][2])))
                    elif type(gene[_property][0]) is str:
                        gene_with_random_params[_property] = str(np.random.choice(gene[_property]))
                break
        return gene_with_random_params

    ##################################################################################
    # Crossover
    ##################################################################################
    def crossover(self, path, fittest_chromosomes):
        # generate N evenly spaced numbers (+ one more because we want to omit 0 afterwards)
        choice_probabilities = np.linspace(0, 1, len(fittest_chromosomes) + 1)[1:][::-1]
        # divide the generated numbers by their sum, to get values between 0 and 1 that sum up to 1
        # --> this is equal to a CDF from a uniform distribution
        choice_probabilities = choice_probabilities / np.sum(choice_probabilities)

        new_population = []
        while len(new_population) < self.params['population_size']:
            # get random chromosome
            chromosome_1_name = np.random.choice(fittest_chromosomes, p=choice_probabilities)

            # get another random chromosome (make sure to not take the same chromosome again)
            chromosome_2_name = chromosome_1_name
            while chromosome_1_name == chromosome_2_name:
                chromosome_2_name = np.random.choice(fittest_chromosomes, p=choice_probabilities)

            # load chromosomes
            with open(path + chromosome_1_name + '/chromosome.json') as f:
                chromosome_1 = json.loads(f.read())
            with open(path + chromosome_2_name + '/chromosome.json') as f:
                chromosome_2 = json.loads(f.read())

            new_chromosome_1, new_chromosome_2 = self._crossover_chromosomes(chromosome_1, chromosome_2)

            if new_chromosome_1 is not None and new_chromosome_2 is not None:
                new_population.append(new_chromosome_1)
                new_population.append(new_chromosome_2)
        return new_population

    def _crossover_chromosomes(self, chromosome_1, chromosome_2):
        # get the indices where preprocessing ends and where the classification layers start
        # --> between those layers we will determine a random crossover point
        idx_start_1 = self._get_first_conv_layer_index(chromosome_1)
        idx_start_2 = self._get_first_conv_layer_index(chromosome_2)
        idx_end_1 = self._get_flatten_gap_gmp_index(chromosome_1)
        idx_end_2 = self._get_flatten_gap_gmp_index(chromosome_2)

        # get two random split points
        chr_1_split = np.random.randint(idx_start_1, idx_end_1)
        chr_2_split = np.random.randint(idx_start_2, idx_end_2)

        new_chromosome_1, new_chromosome_2 = None, None
        i = 0
        while i < 100:
            if chromosome_1[chr_1_split] in self._get_possible_genes(chromosome_2[chr_2_split + 1]) \
                    and chromosome_2[chr_2_split] in self._get_possible_genes(chromosome_1[chr_1_split + 1]):

                new_chromosome_1 = chromosome_1[:chr_1_split:] + chromosome_2[chr_2_split::]

                break
            i += 1

        return new_chromosome_1, new_chromosome_2

    @staticmethod
    def _get_first_conv_layer_index(chromosome):
        """ Iterate over all genes and return the index where layer C or DC is. """
        for idx, gene in enumerate(chromosome):
            if gene['layer'] == 'C' or gene['layer'] == 'DC':
                return idx

    @staticmethod
    def _get_flatten_gap_gmp_index(chromosome):
        """ Iterate over all genes and return the index where layer F, GAP or GMP is. """
        for idx, gene in enumerate(chromosome):
            if gene['layer'] == 'F' or gene['layer'] == 'GAP' or gene['layer'] == 'GMP':
                return idx

    ##################################################################################
    # Mutation
    ##################################################################################
    def mutate_chromosome(self, chromosome):
        mutations = ['drop', 'add', 'params']
        mutation_probability = self.params['mutation_rate']

        idx = 0
        len_chromosome = len(chromosome)
        while idx < len_chromosome:
            if np.random.randint(0, 100) <= mutation_probability:
                mutation = np.random.choice(mutations)
                if mutation == 'drop':
                    if idx == 0 or idx == len(chromosome) - 1:
                        result = self._drop_gene(None, chromosome[idx], None)
                    else:
                        result = self._drop_gene(chromosome[idx - 1], chromosome[idx], chromosome[idx + 1])

                    if result == 'drop':
                        print(f"MUTATION: Removed Layer: {chromosome[idx]['f_name']}")
                        len_chromosome -= 1
                        del chromosome[idx]
                        continue
                    else:  # in this case the gene can't be dropped because of resulting rule set violation
                        pass
                elif mutation == 'add':
                    if idx == 0:
                        gene_to_add = self._get_gene_to_add(None, None)
                        chromosome = self._add_gene(chromosome, gene_to_add, idx)
                        idx += 1
                        continue
                    elif idx + 1 == len_chromosome:
                        gene_to_add = self._get_gene_to_add(chromosome[idx], None)
                    else:
                        gene_to_add = self._get_gene_to_add(chromosome[idx], chromosome[idx + 1])

                    if gene_to_add is not None:
                        print(f"MUTATION: Added Layer: {gene_to_add['f_name']}")
                        chromosome = self._add_gene(chromosome, gene_to_add, idx + 1)
                        idx += 1
                elif mutation == 'params':
                    print(f"MUTATION: Mutated Layer: {chromosome[idx]['f_name']}")
                    mutated_gene = self._mutate_parameters(chromosome[idx])
                    chromosome = self._replace_gene(chromosome, mutated_gene, idx)

            idx += 1

        return chromosome

    def _drop_gene(self, previous_gene, current_gene, following_gene):
        # this means that the first or last layer is affected --> they can be dropped always
        if previous_gene is None or following_gene is None:
            return 'drop'

        # don't drop GMP, GAP or Flatten layer
        if current_gene['layer'] == 'GMP' or current_gene['layer'] == 'GAP' or current_gene['layer'] == 'F':
            return None

        # check if dropping the layer violates the rule set
        rule_set_is_violated = self._check_rule_set_violation(previous_gene, following_gene)
        if rule_set_is_violated:
            return None

        return 'drop'

    def _get_gene_to_add(self, current_gene, following_gene):
        if current_gene is None:
            return self._get_gene_with_random_parameters(np.random.choice(['C', 'DC']))

        # get all possible genes that can follow the current gene and select one randomly
        random_gene = np.random.choice(self._get_possible_genes(current_gene['layer']))

        # get again all possible genes that can follow after the randomly selected gene
        possible_genes = self._get_possible_genes(random_gene)

        # check if the following gene is allowed (if it is None, that means that the random gene will be the last layer)
        if following_gene is None or following_gene['layer'] in possible_genes:
            return self._get_gene_with_random_parameters(random_gene)
        else:
            return None

    @staticmethod
    def _add_gene(chromosome, new_gene, pos):
        new_chromosome = []
        idx = 0
        while idx < len(chromosome):
            if idx == pos:
                new_chromosome.append(new_gene)
            new_chromosome.append(chromosome[idx])
            idx += 1
        return new_chromosome

    @staticmethod
    def _replace_gene(chromosome, mutated_gene, pos):
        return [mutated_gene if idx == pos else gene for idx, gene in enumerate(chromosome)]

    def _mutate_parameters(self, current_gene):
        mutated_gene = self._get_gene_with_random_parameters(current_gene['layer'])
        return mutated_gene

    ##################################################################################
    # Helper
    ##################################################################################
    def _check_rule_set_violation(self, first_gene, second_gene):
        """
        Method that checks if the second given gene can follow after the first one.
        @return: True, if the rule set is violated. False, if the rule set is not violated.
        """
        possible_genes = self._get_possible_genes(first_gene['layer'])
        if second_gene['layer'] not in possible_genes:
            return True
        else:
            return False

    def _get_possible_genes(self, previous_layer):
        """
        Method to apply the previously defined rule set (rule_set.txt) given the previous layer.
        """
        if type(previous_layer) == dict:
            raise ValueError(f"Parameter 'previous_layer' has to be a layer abbreviation like 'C' or 'GMP'. "
                             f"Received {previous_layer} instead.")

        for r in self.rule_set:
            if r['layer'] == previous_layer:
                if 'only_allowed' in r.keys():
                    return r['only_allowed']
                elif 'not_allowed' in r.keys():
                    not_allowed = [d['not_allowed'] for d in self.rule_set if d['layer'] == previous_layer][0]
                    return [d['layer'] for d in self.gene_pool if d['layer'] not in not_allowed]

        # return all layers if there is no entry in the rule set
        return [d['layer'] for d in self.gene_pool]
