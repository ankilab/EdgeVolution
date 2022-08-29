from ast import literal_eval
import numpy as np


class GenePool:
    def __init__(self, params):
        self.params = params
        with open(params['path_gene_pool'], "r") as f:
            self.gene_pool = literal_eval(f.read())

        with open(params['path_rule_set'], "r") as f:
            self.rule_set = literal_eval(f.read())

    def get_random_chromosome(self):
        # create a gene sequence containing all layers of this random gene
        # first layer is always 'C' or 'DC'
        current_gene = np.random.choice(['C', 'DC'])
        chromosome = [self._get_gene_with_random_parameters(current_gene)]

        for _ in range(1, np.random.choice(np.arange(1, self.params['max_nb_feature_layers'] + 1))):
            possible_gene = self._get_possible_genes(current_gene)  # check rule set
            next_gene = np.random.choice(possible_gene)
            current_gene = next_gene

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(current_gene))

        next_gene = np.random.choice(['GAP', 'GMP', 'F'])
        current_gene = next_gene
        chromosome.append(self._get_gene_with_random_parameters(current_gene))

        for _ in range(1, np.random.choice(np.arange(1, self.params['max_nb_classification_layers'] + 1))):
            possible_gene = self._get_possible_genes(current_gene)  # check rule set
            next_gene = np.random.choice(possible_gene)
            current_gene = next_gene

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(current_gene))

        return chromosome

    def _get_gene_with_random_parameters(self, target_gene: str) -> dict:
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

    def mutate_chromosome(self, chromosome):
        mutations = ['drop', 'add']#, 'params']
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
                    chromosome.insert(idx, mutated_gene)

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
        possible_genes = self._get_possible_genes(previous_gene['layer'])
        if following_gene['layer'] not in possible_genes:
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

    def _mutate_parameters(self, current_gene):
        for gene in self.gene_pool:
            if gene == current_gene:
                pass

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
