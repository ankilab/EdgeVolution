"""
This script contains the GenePool class, which is responsible for creating, mutating and crossing over chromosomes. It only belongs to
the implementation of the evolutionary/genetic algorithm.
"""

import logging
import numpy as np
import json
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class GenePool:
    def __init__(self, cfg: DictConfig):
        self.params = cfg.hyperparameters

        search_space = cfg.search_space

        if 'gene_pool' in search_space:
            # Legacy format
            self.gene_pool = [gene for group in search_space.gene_pool.values() for gene in group]
            self.rule_set = [{'layer': key, 'rule': search_space.rule_set[key]['rule']} for key in search_space.rule_set.keys()]
        elif 'layers' in search_space:
            # Unified format: convert to legacy structures
            self.gene_pool, self.rule_set = self._convert_unified_format(search_space)
        else:
            raise ValueError("Search space config must contain 'gene_pool' or 'layers'.")

    @staticmethod
    def _convert_unified_format(search_space):
        """Convert unified format (categories/layers/start) to legacy gene_pool and rule_set."""
        categories = dict(search_space.get('categories', {}))
        layers_cfg = dict(search_space.get('layers', {}))
        start_layers = list(search_space.get('start', []))

        # Build gene_pool: flat list of gene dicts
        gene_pool = []
        for layer_name, layer_def in layers_cfg.items():
            layer_def = dict(layer_def)
            gene = {'layer': layer_name, 'f_name': layer_def.get('f_name', layer_name)}
            for key, value in layer_def.items():
                if key in ('category', 'f_name', 'successors', 'terminal'):
                    continue
                # Convert to list if not already (OmegaConf ListConfig → list)
                gene[key] = list(value) if hasattr(value, '__iter__') and not isinstance(value, str) else value
            gene_pool.append(gene)

        # Build mapping: layer_name → category
        layer_to_category = {}
        for layer_name, layer_def in layers_cfg.items():
            layer_def = dict(layer_def)
            layer_to_category[layer_name] = layer_def.get('category', '')

        # Collect all layers belonging to each category
        category_layers = {}
        for layer_name, cat in layer_to_category.items():
            category_layers.setdefault(cat, []).append(layer_name)

        def resolve_successors(successor_list):
            """Resolve a successor list (can contain category names or layer names)."""
            resolved = []
            for item in successor_list:
                if item in categories:
                    resolved.extend(category_layers.get(item, []))
                elif item in layers_cfg:
                    resolved.append(item)
            return resolved

        # Build rule_set
        rule_set = [{'layer': 'Start', 'rule': start_layers}]
        for layer_name, layer_def in layers_cfg.items():
            layer_def = dict(layer_def)
            if 'successors' in layer_def:
                successors = resolve_successors(list(layer_def['successors']))
            else:
                cat = layer_def.get('category', '')
                cat_cfg = categories.get(cat, {})
                cat_successors = list(cat_cfg.get('successors', []))
                successors = resolve_successors(cat_successors)
            rule_set.append({'layer': layer_name, 'rule': successors})

        return gene_pool, rule_set

    def create_gene_sequence(self):
        """ Create a gene sequence containing all layers of this random gene. """

        chromosome = []

        # get one of the possible starting genes (which are defined in rule_set.txt)
        gene = self._get_start_gene()
        chromosome.append(self._get_gene_with_random_parameters(gene))

        num_feature_layers = np.random.randint(5, self.params.max_num_feature_layers.value + 1)
        # Layers that should not be picked during the feature extraction loop
        # (global pooling and classification are added explicitly afterwards)
        _skip_in_feature_loop = {'GAP_1D', 'GAP_2D', 'GMP_1D', 'GMP_2D', 'FLAT', 'D'}
        for _ in range(num_feature_layers):
            possible_genes = self._get_possible_genes(gene)  # check rule set
            # Filter out global pooling / classification layers
            feature_genes = [g for g in possible_genes if g not in _skip_in_feature_loop]
            if not feature_genes:
                break
            gene = np.random.choice(feature_genes)

            # add current layer
            chromosome.append(self._get_gene_with_random_parameters(gene))

        # check if the last layer at this point is one with complex output (i.e., STFT)
        # --> is yes, add a Magnitude layer
        if 'STFT_2D' in chromosome[-1]['layer']:
            chromosome.append(self._get_gene_with_random_parameters('MAG_2D'))

        # Determine 1D vs 2D by scanning chromosome backwards for a layer with a suffix
        dimensionality = None
        for g in reversed(chromosome):
            if '2D' in g['layer']:
                dimensionality = '2D'
                break
            elif '1D' in g['layer']:
                dimensionality = '1D'
                break

        if dimensionality == '2D':
            gene = np.random.choice(['GAP_2D', 'GMP_2D'])
        elif dimensionality == '1D':
            gene = np.random.choice(['GAP_1D', 'GMP_1D'])
        else:
            raise RuntimeError("Couldn't determine if the architecture is 1D or 2D.")
        chromosome.append(self._get_gene_with_random_parameters(gene))

        num_classification_layers = np.random.randint(3, self.params.max_num_classification_layers.value + 1)
        for _ in range(num_classification_layers):
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
            # Uniform selection (probability-weighted selection was tested but
            # showed no improvement over uniform for this problem)
            chromosome_1_name = np.random.choice(fittest_chromosomes)

            # get another random chromosome (make sure to not take the same chromosome again)
            chromosome_2_name = chromosome_1_name
            while chromosome_1_name == chromosome_2_name:
                chromosome_2_name = np.random.choice(fittest_chromosomes)

            # load chromosomes
            with open(path + chromosome_1_name + '/chromosome.json') as f:
                chromosome_1 = json.loads(f.read())
            with open(path + chromosome_2_name + '/chromosome.json') as f:
                chromosome_2 = json.loads(f.read())

            try:
                new_chromosomes, chr_1_split, chr_2_split = self._crossover_chromosomes(chromosome_1, chromosome_2)
            except Exception as e:
                logger.debug("Crossover failed: %s", e)
                continue

            if new_chromosomes is not None:
                for new_chromosome in new_chromosomes:
                    if new_chromosome is not None:
                        new_population.append(new_chromosome)

                        # save both parents names to be able to follow the whole evolutionary process later
                        parents_names.append((chromosome_1_name, chromosome_2_name, chr_1_split, chr_2_split))

        return new_population, parents_names

    def crossover_from_chromosomes(self, fittest_dict):
        """Crossover using chromosomes from a dict instead of reading from disk.

        Args:
            fittest_dict: Dict mapping name -> chromosome (list of gene dicts).

        Returns:
            (new_population, parents_names) same as crossover().
        """
        fittest_names = list(fittest_dict.keys())

        new_population = []
        parents_names = []
        while len(new_population) < self.params.population_size.value:
            chromosome_1_name = np.random.choice(fittest_names)

            chromosome_2_name = chromosome_1_name
            while chromosome_1_name == chromosome_2_name:
                chromosome_2_name = np.random.choice(fittest_names)

            chromosome_1 = fittest_dict[chromosome_1_name]
            chromosome_2 = fittest_dict[chromosome_2_name]

            try:
                new_chromosomes, chr_1_split, chr_2_split = self._crossover_chromosomes(
                    chromosome_1, chromosome_2
                )
            except Exception as e:
                logger.debug("Crossover failed: %s", e)
                continue

            if new_chromosomes is not None:
                for new_chromosome in new_chromosomes:
                    if new_chromosome is not None:
                        new_population.append(new_chromosome)
                        parents_names.append(
                            (chromosome_1_name, chromosome_2_name, chr_1_split, chr_2_split)
                        )

        return new_population, parents_names

    def _crossover_chromosomes(self, chromosome_1, chromosome_2):
        # Crossover point is chosen between the first layer and the pooling layer
        idx_start_1, idx_start_2 = 1, 1
        idx_end_1 = self._get_flatten_gap_gmp_index(chromosome_1)
        idx_end_2 = self._get_flatten_gap_gmp_index(chromosome_2)

        # get two random split points
        if idx_start_1 is None or idx_end_1 is None:
            logger.debug("No valid crossover region in chromosome 1")
            return None, None, None
        if idx_start_2 is None or idx_end_2 is None:
            logger.debug("No valid crossover region in chromosome 2")
            return None, None, None

        chr_1_split = np.random.randint(idx_start_1, idx_end_1)
        chr_2_split = np.random.randint(idx_start_2, idx_end_2)

        i = 0
        while True:
            i += 1
            rule_set_is_violated_1 = self._check_rule_set_violation(chromosome_1[chr_1_split], chromosome_2[chr_2_split + 1])
            rule_set_is_violated_2 = self._check_rule_set_violation(chromosome_2[chr_2_split], chromosome_1[chr_1_split + 1])
            if not rule_set_is_violated_1 and not rule_set_is_violated_2:
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
            logger.debug("Changed GAP/GMP to 2D")
        elif '1D' in chromosome[idx_gap_gmp - 1]['layer'] and '2D' in chromosome[idx_gap_gmp]['layer']:
            chromosome[idx_gap_gmp]['layer'] = chromosome[idx_gap_gmp]['layer'].replace('2D', '1D')
            chromosome[idx_gap_gmp]['f_name'] = chromosome[idx_gap_gmp]['f_name'].replace('2D', '1D')
            logger.debug("Changed GAP/GMP to 1D")

        return chromosome

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
                        logger.debug(f"MUTATION: Removed Layer: {chromosome[idx]['f_name']}")
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
                        logger.debug(f"MUTATION: Added Layer: {gene_to_add['f_name']}")
                        chromosome = self._add_gene(chromosome, gene_to_add, idx + 1)
                        idx += 1
                elif mutation == 'params' and idx != 0:
                    logger.debug(f"MUTATION: Mutated Layer: {chromosome[idx]['f_name']}")
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
        Method to apply the previously defined rule set (search_space.yaml) to find a layer
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
