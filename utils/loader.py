from pathlib import Path
import json
import os
from omegaconf import DictConfig

class Loader:
    def __init__(self, continue_path: str, continue_generation: int):
        self.generation = continue_generation
        self.ga_path = Path(continue_path)
        self.gen_path = self.ga_path / f"Generation_{self.generation}"

        self.cfg = None

    def get_cfg(self):
        with open(self.ga_path / 'config.json') as f:
            self.cfg = json.loads(f.read())
            
        with open(self.ga_path / 'search_space.json') as f:
            self.cfg["search_space"] = json.loads(f.read())
        
        # add information from which run the current run was continued
        self.cfg["continued_from"] = str(self.gen_path)
        self.cfg = DictConfig(self.cfg)
        return self.cfg

    def load_population_genotype(self):
        population_genotype = []
        population = [ind for ind in os.listdir(self.gen_path)]
        population.sort()

        for ind in population:
            with open(self.gen_path / str(ind) / 'chromosome.json') as f:
                chromosome = json.loads(f.read())
            population_genotype.append(chromosome)
        return population_genotype

    def load_individuals(self):
        individuals = {}
        population = [ind for ind in os.listdir(self.gen_path)]
        population.sort()

        for individual in population:
            with open(self.gen_path / str(individual) / 'chromosome.json') as f:
                chromosome = json.loads(f.read())
            individuals[individual] = {'genotype': chromosome}
        return individuals
