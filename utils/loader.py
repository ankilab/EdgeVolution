

class Loader:
    def __init__(self, continue_from: dict):
        self.ga_run = continue_from['continue_from_ga_run']
        self.generation = continue_from['continue_from_generation']

    def get_params(self):
        pass

    def get_gen_individuals(self):
        pass

    def get_gen_start(self) -> int:
        return self.generation
