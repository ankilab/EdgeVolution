import os
import time
import json
import csv
import subprocess
import shutil
from pathlib import Path
import git
from omegaconf import OmegaConf, DictConfig


class Saver:
    def __init__(self, experiment):
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.results_dir = Path(f"Results/evonas_{time.strftime('%Y%m%d-%H%M%S')}_{experiment}")
        os.mkdir(self.results_dir)

        self.random_names = []

    def save_params(self, cfg: DictConfig):
        # get git commit hash and add it to params
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        cfg.hyperparameters.git_sha.value = sha

        with open(self.results_dir / "config.json", "w") as f:
            cfg_dict = OmegaConf.to_container(cfg)
            cfg_dict = {k: cfg_dict[k] for k in ['boards', 'hyperparameters', 'results', 'fitness_function']}
            json.dump(cfg_dict, f, indent=4)

        # save search_space.json
        with open(self.results_dir / "search_space.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg.search_space), f, indent=4)
        

    def _get_path(self, gen_count, name=None):
        if name is None:
            return self.results_dir / f'Generation_{gen_count}'
        else:
            return self.results_dir / f'Generation_{gen_count}/{name}'

    @staticmethod
    def _save_chromosome_genotype(chromosome, path):
        with open(path / 'chromosome.json', 'w') as f:
            json.dump(chromosome, f, indent=2)

    def save_best_individual(self, gen_count: int, name: str, fitness: float):
        row = [f'Generation_{gen_count}', name, fitness]
        with open(self.results_dir / r'best_individual_each_generation.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_parents(self, gen_count, individuals_new_generation, parents_names):
        for individual, (parent_1, parent_2, parent_1_split, parent_2_split) in zip(individuals_new_generation, parents_names):
            # parent_1_split and parent_2_split are the idx where the chromosomes are split up
            row = [f'Generation: {gen_count}', f'Parent_1: ({parent_1}, {parent_1_split})',
                   f'Parent_2: ({parent_2}, {parent_2_split})', f'New_Individual: {individual}']
            with open(self.results_dir / r'crossover_parents.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

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
            p = p / "models"
            os.mkdir(p)
            self._save_chromosome_phenotype(chromosome_p, chromosome_p_tflite, p)

            # convert tflite model to C-array
            subprocess.call("xxd -i " + str(p / "model_tflite_untrained.tflite") + " > " + str(p / "model_c_array_untrained.cc"), shell=True)

    def create_generation_dir(self, individuals: dict, gen_count: int) -> None:
        # create generation dir
        os.mkdir(self._get_path(gen_count))

        # generate individual dir (without chromosome)
        for name in individuals.keys():
            p = self._get_path(gen_count, name)
            os.mkdir(p)

    def save_population_genotype(self, individuals: dict, gen_count) -> None:
        for name in individuals.keys():
            p = self._get_path(gen_count, name)
            self._save_chromosome_genotype(individuals[name]['genotype'], p)

    def save_population_phenotype(self, name: str, gen_count, model_untrained) -> None:
        p = self._get_path(gen_count, name)
        p = p / "models"
        os.mkdir(p)

        model_untrained.save(p / "model_untrained.h5")

    def save_population_phenotype_tflite(self, name: str, gen_count, model_tflite_untrained) -> None:
        p = self._get_path(gen_count, name)
        p = p / "models"
        if not os.path.exists(p):
            os.mkdir(p)

        with open(p / 'model_tflite_untrained.tflite', 'wb') as f:
            f.write(model_tflite_untrained)

        # convert tflite model to C-array
        try:
            subprocess.call("xxd -i " + str(p / "model_tflite_untrained.tflite") + " > " + str(p / "model_c_array_untrained.cc"), shell=True, timeout=20)
        except subprocess.TimeoutExpired:
            print("xxd command timed out")



