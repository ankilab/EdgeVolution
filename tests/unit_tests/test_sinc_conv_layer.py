import unittest
import tensorflow as tf
import numpy as np

from genetic_algorithm.src.genepool_modules.sinc_conv_layer import sinc, generate_sinc_kernel, SincConv1D


class TestSincConv1D(unittest.TestCase):

    #  Tests that the layer can be initialized with the correct parameters
    def test_initialization_with_correct_parameters(self):
        num_filters = 16
        kernel_size = 16
        cutoff = 4

        layer = SincConv1D(num_filters, kernel_size, cutoff)

        self.assertEqual(layer.num_filters, num_filters)
        self.assertEqual(layer.kernel_size, kernel_size)
        self.assertEqual(layer.cutoff, cutoff)

    #  Tests that the layer can be built with the correct input shape
    def test_building_with_correct_input_shape(self):
        num_filters = 16
        kernel_size = 16
        cutoff = 4
        input_shape = (None, 100, 1)

        layer = SincConv1D(num_filters, kernel_size, cutoff)
        layer.build(input_shape)

        self.assertEqual(layer.kernel.shape, (kernel_size, 1, num_filters))

    #  Tests that the layer returns an output tensor with the correct shape
    def test_returning_output_tensor_with_correct_shape(self):
        num_filters = 16
        kernel_size = 16
        cutoff = 4
        input_shape = (1, 100, 1)
        input_tensor = tf.random.normal(input_shape)

        expected_output_shape = (1, 100, num_filters)

        layer = SincConv1D(num_filters, kernel_size, cutoff)
        layer.build(input_shape)
        output_tensor = layer.call(input_tensor)

        self.assertEqual(output_tensor.shape, expected_output_shape)


class TestGenerateSincKernel(unittest.TestCase):

    #  Tests that the function generates a kernel of size 10 and cutoff 2
    def test_generate_kernel_size_10_cutoff_2(self):
        kernel = generate_sinc_kernel(10, 2)
        self.assertEqual(len(kernel), 10)
        self.assertEqual(kernel[0], kernel[-1])
        self.assertAlmostEqual(np.sum(kernel), 1.0)

    #  Tests that the function generates a kernel of size 5 and cutoff 1
    def test_generate_kernel_size_5_cutoff_1(self):
        kernel = generate_sinc_kernel(5, 1)
        self.assertEqual(len(kernel), 5)
        self.assertEqual(kernel[0], kernel[-1])
        self.assertAlmostEqual(np.sum(kernel), 1.0)

    #  Tests that the function generates a kernel of size 10 and cutoff 5 with all zeros except for the center value
    def test_generate_kernel_size_10_cutoff_5(self):
        kernel = generate_sinc_kernel(10, 5)
        self.assertEqual(len(kernel), 10)
        self.assertEqual(kernel[0], kernel[-1])
        self.assertAlmostEqual(np.sum(kernel), 1.0)


class TestSinc(unittest.TestCase):

    #  Tests that the function returns 1.0 when the input array contains a single zero element
    def test_zero_input(self):
        input_array = np.array([0])
        expected_output = 1.0
        self.assertEqual(sinc(input_array), expected_output)

    #  Tests that the function returns the correct sinc value for a positive input value
    def test_positive_input(self):
        input_array = np.array([1])
        expected_output = np.sin(np.pi) / (np.pi)
        self.assertEqual(sinc(input_array), expected_output)

    #  Tests that the function returns the correct sinc value for a negative input value
    def test_negative_input(self):
        input_array = np.array([-1])
        expected_output = np.sin(-np.pi) / (-np.pi)
        self.assertEqual(sinc(input_array), expected_output)

    #  Tests that the function returns an array of ones when the input array contains all zero elements
    def test_zero_array(self):
        input_array = np.array([0, 0, 0])
        expected_output = np.array([1, 1, 1])
        np.testing.assert_array_equal(sinc(input_array), expected_output)

    #  Tests that the function returns the correct sinc value for a large input value
    def test_large_input(self):
        input_array = np.array([100])
        expected_output = np.sin(np.pi * 100) / (np.pi * 100)
        self.assertEqual(sinc(input_array), expected_output)

    #  Tests that the function returns the correct sinc value for a small input value
    def test_small_input(self):
        input_array = np.array([0.01])
        expected_output = np.sin(np.pi * 0.01) / (np.pi * 0.01)
        self.assertEqual(sinc(input_array), expected_output)
