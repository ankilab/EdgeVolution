import unittest
from genetic_algorithm.src.genepool import GenePool

params = {"path_gene_pool": "gene_pool.txt", "path_rule_set": "rule_set.txt", "max_nb_feature_layers": 30,
          "max_nb_classification_layers": 10}
gene_pool = GenePool(params)


class TestGenePool(unittest.TestCase):
    ###################################################################
    # Tests for rule set
    ###################################################################
    def test_get_random_gene_sequence(self):
        gene_sequence = gene_pool.create_gene_sequence()
        self.assertIsNotNone(gene_sequence)

    def test_get_possible_layers_only_allowed_GAP(self):
        _input = 'GAP_2D'
        expected = ['D', 'DO']
        actual = gene_pool._get_possible_genes(_input)
        self.assertListEqual(expected, actual)

    def test_get_possible_layers_only_allowed_BN(self):
        _input = 'BN_2D'
        expected = ['AP_2D', 'MP_2D', 'R_2D', 'DC_2D', 'C_2D']
        actual = gene_pool._get_possible_genes(_input)
        self.assertListEqual(expected, actual)

    def test_add_gene(self):
        # Test Case 1
        _input_1 = {'layer': 'BN_2D'}
        _input_2 = {'layer': 'C_2D'}
        expected = ['AP_2D', 'MP_2D', 'R_2D', 'DC_2D', 'C_2D']
        actual = gene_pool._get_gene_to_add(_input_1, _input_2)
        self.assertIn(actual['layer'], expected)

        # Test Case 2

        # IN_2D removed
        
        # _input_1 = {'layer': 'IN_2D'}
        # _input_2 = None
        # expected = ['AP_2D', 'MP_2D', 'R_2D', 'DC_2D', 'C_2D']
        # actual = gene_pool._get_gene_to_add(_input_1, _input_2)
        # self.assertIn(actual['layer'], expected)

    def test_drop_gene(self):
        # Test Case 1
        _input_1 = {'layer': 'BN_2D'}
        _input_2 = {'layer': 'C_2D'}
        _input_3 = {'layer': 'BN_2D'}
        expected = None
        actual = gene_pool._drop_gene(_input_1, _input_2, _input_3)
        self.assertEqual(actual, expected)

        # Test Case 2
        _input_1 = {'layer': 'C_2D'}
        _input_2 = {'layer': 'BN_2D'}
        _input_3 = {'layer': 'R_2D'}
        expected = 'drop'
        actual = gene_pool._drop_gene(_input_1, _input_2, _input_3)
        self.assertEqual(actual, expected)

    def test_replace_gene(self):
        # Test Case 1
        _input_1 = [{'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'R_2D'}]
        _input_2 = {'layer': 'IN_2D'}
        _input_3 = 1
        expected = [{'layer': 'C_2D'}, {'layer': 'IN_2D'}, {'layer': 'R_2D'}]
        actual = gene_pool._replace_gene(_input_1, _input_2, _input_3)
        self.assertEqual(actual, expected)

        # Test Case 2
        _input_1 = [{'layer': 'C_2D'}, {'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'R_2D'}]
        _input_2 = {'layer': 'AP_2D'}
        _input_3 = 3
        expected = [{'layer': 'C_2D'}, {'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'AP_2D'}]
        actual = gene_pool._replace_gene(_input_1, _input_2, _input_3)
        self.assertEqual(actual, expected)

    def test_crossover_chromosome(self):
        # Test Case 1
        _input_1 = [{'layer': 'STFT'}, {'layer': 'MAG'}, {'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'R_2D'},
                    {'layer': 'GAP_2D'}, {'layer': 'D'}]
        _input_2 = [{'layer': 'STFT'}, {'layer': 'MAG'}, {'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'R_2D'},
                    {'layer': 'GAP_2D'}, {'layer': 'D'}]

        expected_1 = [{'layer': 'STFT'}, {'layer': 'MAG'}, {'layer': 'C_2D'}, {'layer': 'BN_2D'}, {'layer': 'R_2D'},
                      {'layer': 'GAP_2D'}, {'layer': 'D'}]
        expected_2 = [{'layer': 'STFT'}, {'layer': 'MAG'}, {'layer': 'C_2D'},  {'layer': 'R_2D'},
                      {'layer': 'GAP_2D'}, {'layer': 'D'}]
        actual, _, _ = gene_pool._crossover_chromosomes(_input_1, _input_2)
        self.assertTrue(actual == expected_1 or actual == expected_2)


if __name__ == '__main__':
    unittest.main()
