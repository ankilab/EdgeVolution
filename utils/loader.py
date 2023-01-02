from pathlib import Path
import json
import os


class Loader:
    def __init__(self, continue_from: dict):
        self.ga_run = continue_from['continue_from_ga_run']
        self.generation = continue_from['continue_from_generation']
        self.ga_path = Path("Results/" + self.ga_run)
        self.gen_path = self.ga_path / f"Generation_{self.generation}"

        self.params = None

    def get_params(self):
        with open(self.ga_path / 'params.json') as f:
            self.params = json.loads(f.read())
        
        # add information from which run the current run was continued
        self.params["continued_from"] = str(self.gen_path)
        return self.params

    def load_population_genotype(self):
        population_genotype = []
        for ind in os.listdir(self.gen_path):
            with open(self.gen_path / str(ind) / 'chromosome.json') as f:
                chromosome = json.loads(f.read())
            population_genotype.append(chromosome)
        return population_genotype

    def get_gen_start(self) -> int:
        return int(self.generation)
