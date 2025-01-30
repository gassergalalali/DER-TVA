"""
plant.py

Plants are just a group of generators...
"""
import typing

from DER.agents.abstract_agent import AbstractAgent
from DER.agents.generator import Generator

class Plant(AbstractAgent):
    """Plants are just a group of generators"""
    def __init__(
        self,
        name: str = None,
        simulation = None
        ) -> None:
        super().__init__()
        self.generators: typing.List[Generator] = []
        self.name: str = name
        self.simulation = simulation
        self.simulation.plants.append(self)

    def attach_generator(self, generator: Generator):
        if generator in self.generators:
            self.logger.error("Generator is already assigned to this plant")
        else:
            self.generators.append(generator)
            generator.plant = self

    def setup(self):
        pass

    def remove(self):
        """Remove self from the simulation"""
        self.simulation.plants.remove(self)
        for generator in self.generators:
            generator.plant = None
