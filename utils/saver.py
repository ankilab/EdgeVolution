import pandas as pd
import os
import time
import tensorflow as tf
import json


class Saver:
    def __init__(self):
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.results_dir = f"Results/ga_{time.strftime('%Y%m%d-%H%M%S')}"
        os.mkdir(self.results_dir)

        self.random_names = []

    def save_params(self, params):
        with open(self.results_dir + '/params.json', 'w') as f:
            json.dump(params, f, indent=4)

    def _get_path(self, gen_count, name=None):
        if name is None:
            return self.results_dir + f'/Generation_{gen_count}'
        else:
            return self.results_dir + f'/Generation_{gen_count}/{name}'

    def save_chromosomes(self, population_genotype: list, population_phenotype: list, population_phenotype_tflite: list,
                         chromosome_names: list, gen_count: int) -> None:
        # create generation dir
        os.mkdir(self._get_path(gen_count))
        # generate individual dir and safe chromosome
        for name, chromosome_g, chromosome_p, chromosome_p_tflite \
                in zip(chromosome_names, population_genotype, population_phenotype, population_phenotype_tflite):

            p = self._get_path(gen_count, name)
            os.mkdir(p)
            self._save_chromosome_genotype(chromosome_g, p)
            p = p + "/models"
            os.mkdir(p)
            self._save_chromosome_phenotype(chromosome_p, chromosome_p_tflite, p)

    def save_phenotypes(self, tf_model):
        pass

    @staticmethod
    def _save_chromosome_genotype(chromosome, path):
        with open(path + '/chromosome.json', 'w') as f:
            json.dump(chromosome, f, indent=2)

    @staticmethod
    def _save_chromosome_phenotype(model_untrained, model_tflite_untrained, path):
        model_untrained.save(path + "/model_untrained.h5")

        with open(path + '/model_tflite_untrained.tflite', 'wb') as f:
            f.write(model_tflite_untrained)

