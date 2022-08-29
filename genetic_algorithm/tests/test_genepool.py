import unittest
from evolutionary_algorithm.src.genepool import GenePool

path_gene_pool = "../gene_pool.txt"
path_rule_set = "../rule_set.txt"

gene_pool = GenePool(path_gene_pool, path_rule_set)


class TestGenePool(unittest.TestCase):
    ###################################################################
    # Tests for rule set
    ###################################################################
    def test_get_random_gene_sequence(self):
        gene_sequence = gene_pool.get_random_gene_sequence()
        self.assertIsNotNone(gene_sequence)

    def test_get_possible_layers_no_rules(self):
        _input = 'C'
        expected = ['C', 'DC', 'MP', 'AP', 'GAP', 'GMP', 'F', 'BN', 'IN', 'R']
        actual = gene_pool._get_possible_layers(_input)
        self.assertListEqual(expected, actual)

    def test_get_possible_layers_not_allowed(self):
        _input = 'AP'
        expected = ['C', 'DC', 'F', 'BN', 'IN', 'R']
        actual = gene_pool._get_possible_layers(_input)
        self.assertListEqual(expected, actual)

    def test_get_possible_layers_only_allowed(self):
        _input = 'F'
        expected = ['D', 'DO']
        actual = gene_pool._get_possible_layers(_input)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
