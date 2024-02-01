from ast import literal_eval
import numpy as np
import json
from omegaconf import DictConfig


class GenePool:
    def __init__(self, cfg: DictConfig):
        self.params = cfg.hyperparameters

        self.gene_pool = [gene for group in cfg.search_space.gene_pool.values() for gene in group]
        
        self.rule_set = [{'layer': key, 'rule': cfg.search_space.rule_set[key]['rule']} for key in cfg.search_space.rule_set.keys()]

    def get_random_chromosome(self):
        """ Create a gene sequence containing all layers of this random gene. """

        # get one of the possible starting genes (which are defined in rule_set.txt)
        gene = self._get_start_gene()
        chromosome = [self._get_gene_with_random_parameters(gene)]

        for _ in range(0, np.random.choice(np.arange(5, self.params.max_num_feature_layers.value + 1)), 1):
            possible_genes = self._get_possible_genes(gene)  # check rule set
            gene = np.random.choice(possible_genes)

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(gene))

        # check if the last layer at this point is one with complex output (i.e., STFT)
        # --> is yes, add a Magnitude layer
        if 'STFT_2D' in chromosome[-1]['layer']:
            chromosome.append(self._get_gene_with_random_parameters('MAG_2D'))

        # check in the previous gene if we have a 1D or 2D network
        if '2D' in gene:
            gene = np.random.choice(['GAP_2D', 'GMP_2D'])
        elif '1D' in gene:
            gene = np.random.choice(['GAP_1D', 'GMP_1D'])
        else:
            raise RuntimeError("Couldn't determine if the architecture is 1D or 2D.")
        chromosome.append(self._get_gene_with_random_parameters(gene))

        for _ in range(0, np.random.choice(np.arange(3, self.params.max_num_classification_layers.value + 1)), 1):
            possible_genes = self._get_possible_genes(gene)  # check rule set
            gene = np.random.choice(possible_genes)

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(gene))

        return chromosome

    def _get_start_gene(self):
        for rule in self.rule_set:
            if rule['layer'] == 'Start':
                return np.random.choice(rule['rule'])

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
        parents_names = []
        while len(new_population) < self.params.population_size.value:
            # get random chromosome
            #chromosome_1_name = np.random.choice(fittest_chromosomes, p=choice_probabilities)
            chromosome_1_name = np.random.choice(fittest_chromosomes)

            # get another random chromosome (make sure to not take the same chromosome again)
            chromosome_2_name = chromosome_1_name
            while chromosome_1_name == chromosome_2_name:
                #chromosome_2_name = np.random.choice(fittest_chromosomes, p=choice_probabilities)
                chromosome_2_name = np.random.choice(fittest_chromosomes)

            # load chromosomes
            with open(path + chromosome_1_name + '/chromosome.json') as f:
                chromosome_1 = json.loads(f.read())
            with open(path + chromosome_2_name + '/chromosome.json') as f:
                chromosome_2 = json.loads(f.read())

            #try:
            new_chromosomes, chr_1_split, chr_2_split = self._crossover_chromosomes(chromosome_1, chromosome_2)
            #except Exception as e:
                #print(e)
                #continue

            for new_chromosome in new_chromosomes:
                if new_chromosome is not None:
                    new_population.append(new_chromosome)

                    # save both parents names to be able to follow the whole evolutionary process later
                    parents_names.append((chromosome_1_name, chromosome_2_name, chr_1_split, chr_2_split))

        return new_population, parents_names

    def _crossover_chromosomes(self, chromosome_1, chromosome_2):
        # get the indices where preprocessing ends and where the classification layers start
        # --> between those layers we will determine a random crossover point
        # idx_start_1 = self._get_first_conv_layer_index(chromosome_1) --> deprecated
        # idx_start_2 = self._get_first_conv_layer_index(chromosome_2) --> deprecated
        idx_start_1, idx_start_2 = 1, 1
        idx_end_1 = self._get_flatten_gap_gmp_index(chromosome_1)
        idx_end_2 = self._get_flatten_gap_gmp_index(chromosome_2)

        # get two random split points
        if idx_start_1 is None or idx_end_1 is None:
            print("None error (chr 1):", chromosome_1)
            return None, None, None
        if idx_start_2 is None or idx_end_2 is None:
            print("None error (chr 2):", chromosome_2)
            return None, None, None

        chr_1_split = np.random.randint(idx_start_1, idx_end_1)
        chr_2_split = np.random.randint(idx_start_2, idx_end_2)

        i = 0
        while True:
            i += 1
            rule_set_is_violated = self._check_rule_set_violation(chromosome_1[chr_1_split], chromosome_2[chr_2_split + 1])
            if not rule_set_is_violated:
                break
            elif i == 1000:  # no split found --> chromosomes are not crossable
                return None, None, None
            else:
                chr_1_split = np.random.randint(idx_start_1, idx_end_1)
                chr_2_split = np.random.randint(idx_start_2, idx_end_2)

        # crossover chromosomes with determined splits
        new_chromosome_1 = chromosome_1[:chr_1_split+1:] + chromosome_2[chr_2_split+1:idx_end_2:]
        new_chromosome_2 = chromosome_2[:chr_2_split+1:] + chromosome_1[chr_1_split+1:idx_end_1:]

        # Randomly choose which end of the chromosome to add to the new chromosome
        if np.random.randint(0, 2) == 0:
            new_chromosome_1 += chromosome_1[idx_end_1::]
            new_chromosome_2 += chromosome_2[idx_end_2::]
        else:
            new_chromosome_1 += chromosome_2[idx_end_2::]
            new_chromosome_2 += chromosome_1[idx_end_1::]

        # check if GAP or GMP is followed by a 1D or 2D layer and change it if necessary
        new_chromosome_1 = self.check_if_GAP_GMP_1D_or_2D(new_chromosome_1)
        new_chromosome_2 = self.check_if_GAP_GMP_1D_or_2D(new_chromosome_2)

        return [new_chromosome_1, new_chromosome_2], chr_1_split, chr_2_split+1

    def check_if_GAP_GMP_1D_or_2D(self, chromosome):
        """ 
        Check if GAP or GMP is followed by a 1D or 2D layer and change it if necessary. 
        """
        idx_gap_gmp = self._get_flatten_gap_gmp_index(chromosome)
        if '2D' in chromosome[idx_gap_gmp - 1]['layer'] and '1D' in chromosome[idx_gap_gmp]['layer']:
            chromosome[idx_gap_gmp]['layer'] = chromosome[idx_gap_gmp]['layer'].replace('1D', '2D')
            chromosome[idx_gap_gmp]['f_name'] = chromosome[idx_gap_gmp]['f_name'].replace('1D', '2D')
            print("Changed GAP/GMP to 2D. Chromosome:", chromosome)
        elif '1D' in chromosome[idx_gap_gmp - 1]['layer'] and '2D' in chromosome[idx_gap_gmp]['layer']:
            chromosome[idx_gap_gmp]['layer'] = chromosome[idx_gap_gmp]['layer'].replace('2D', '1D')
            chromosome[idx_gap_gmp]['f_name'] = chromosome[idx_gap_gmp]['f_name'].replace('2D', '1D')
            print("Changed GAP/GMP to 1D. Chromosome:", chromosome)

        return chromosome

    @staticmethod
    def _get_first_conv_layer_index(chromosome):
        """ Iterate over all genes and return the index where first layer C or DC is. """
        for idx, gene in enumerate(chromosome):
            if 'C' in gene['layer'] or 'DC' in gene['layer']:
                return idx

    @staticmethod
    def _get_flatten_gap_gmp_index(chromosome):
        """ Iterate over all genes and return the index where layer F, GAP or GMP is. """
        for idx, gene in enumerate(chromosome):
            if 'FLAT' in gene['layer'] or 'GAP' in gene['layer'] or 'GMP' in gene['layer']:
                return idx

    ##################################################################################
    # Mutation
    ##################################################################################
    def mutate_chromosome(self, chromosome):
        mutations = ['drop', 'add', 'params']
        mutation_probability = self.params.mutation_rate.value

        idx = 0
        len_chromosome = len(chromosome)
        while idx < len_chromosome:
            if np.random.randint(0, 100) <= mutation_probability:
                mutation = np.random.choice(mutations)
                if mutation == 'drop':
                    previous_gene = chromosome[idx - 1]
                    current_gene = chromosome[idx]
                    following_gene = None  # have to set it to None at this point (have to check if index [idx + 1] is out of range)

                    # check if we have the first gene or the last gene
                    if idx == 0:
                        previous_gene = None
                    elif not idx == len(chromosome) - 1:
                        following_gene = chromosome[idx + 1]

                    result = self._drop_gene(previous_gene, current_gene, following_gene)
                    if result == 'drop':
                        print(f"MUTATION: Removed Layer: {chromosome[idx]['f_name']}")
                        len_chromosome -= 1
                        del chromosome[idx]
                        continue
                    else:  # in this case the gene can't be dropped because of resulting rule set violation
                        pass
                elif mutation == 'add':
                    if idx + 1 == len_chromosome:
                        gene_to_add = self._get_gene_to_add(chromosome[idx], None)
                    else:
                        gene_to_add = self._get_gene_to_add(chromosome[idx], chromosome[idx + 1])

                    if gene_to_add is not None:
                        print(f"MUTATION: Added Layer: {gene_to_add['f_name']}")
                        chromosome = self._add_gene(chromosome, gene_to_add, idx + 1)
                        idx += 1
                elif mutation == 'params' and idx != 0:
                    print(f"MUTATION: Mutated Layer: {chromosome[idx]['f_name']}")
                    mutated_gene = self._mutate_parameters(chromosome[idx])
                    chromosome = self._replace_gene(chromosome, mutated_gene, idx)

            idx += 1

        return chromosome

    def _drop_gene(self, previous_gene, current_gene, following_gene):
        # this means that the first layer is affected --> don't drop it because it contains preprocessing
        if previous_gene is None:
            return None

        # this means that the last layer is affected --> it can be dropped anyway
        if following_gene is None:
            return 'drop'

        # don't drop GMP, GAP or Flatten layer
        if 'GMP' in current_gene['layer'] or 'GAP' in current_gene['layer'] or 'FLAT' in current_gene['layer']:
            return None

        # check if dropping the layer violates the rule set
        rule_set_is_violated = self._check_rule_set_violation(previous_gene, following_gene)
        if rule_set_is_violated:
            return None

        return 'drop'

    def _get_gene_to_add(self, current_gene, following_gene):
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
        new_chromosome = chromosome[:pos] + [new_gene] + chromosome[pos:]
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
        Method to apply the previously defined rule set (rule_set.txt) to find a layer
        that can follow after a given previous layer.
        """
        if isinstance(previous_layer, dict):
            raise ValueError(f"Parameter 'previous_layer' has to be a layer abbreviation like 'C' or 'GMP'. "
                             f"Received {previous_layer} instead.")

        for rule in self.rule_set:
            if rule['layer'] == previous_layer:
                return rule.get('rule', [])

        # return all layers of the gene pool if there is no entry in the rule set
        return [g['layer'] for g in self.gene_pool]
