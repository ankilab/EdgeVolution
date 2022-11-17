import os
import time
import json
import csv
import subprocess


class Saver:
    def __init__(self, experiment):
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.results_dir = f"Results/ga_{time.strftime('%Y%m%d-%H%M%S')}_{experiment}"
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

            # convert tflite model to C-array
            subprocess.call("xxd -i " + p + "/model_tflite_untrained.tflite > " + p + "/model_c_array_untrained.cc", shell=True)

    @staticmethod
    def _save_chromosome_genotype(chromosome, path):
        with open(path + '/chromosome.json', 'w') as f:
            json.dump(chromosome, f, indent=2)

    @staticmethod
    def _save_chromosome_phenotype(model_untrained, model_tflite_untrained, path):
        model_untrained.save(path + "/model_untrained.h5")

        with open(path + '/model_tflite_untrained.tflite', 'wb') as f:
            f.write(model_tflite_untrained)

    def save_best_individual(self, gen_count, best_individual):
        row = [f'Generation: {gen_count}', best_individual[0], best_individual[1]]
        with open(self.results_dir + r'/best_individual_each_generation.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_parents(self, gen_count, individuals_new_generation, parents_names):
        for individual, (parent_1, parent_2, parent_1_split, parent_2_split) in zip(individuals_new_generation, parents_names):
            # parent_1_split and parent_2_split are the idx where the chromosomes are split up
            row = [f'Generation: {gen_count}', f'Parent_1: ({parent_1}, {parent_1_split})',
                   f'Parent_2: ({parent_2}, {parent_2_split})', f'New_Individual: {individual}']
            with open(self.results_dir + r'/crossover_parents.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)


