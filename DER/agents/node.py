"""
node.py

A node is a Bus in the grid.
Each node has a generator and a consumer base.
"""

import numpy as np

from DER.agents.abstract_agent import AbstractAgent

class Node(AbstractAgent):
    def __init__(
            self,
            simulation,
            name: str = None
    ):
        super().__init__()
        # Record the simulation
        self.simulation = simulation
        self.simulation.nodes.append(self)
        # Variables - Agents
        self.lines = []  # The lines connecting to or from this node.
        self.generators = []  # Will be assigned by the generator object
        self.lses = []  # Will be assigned by the lse object
        # Other variables
        self.name = name if name else f"Node{self.simulation.nodes.index(self)+1}"
        self.lmp: np.ndarray = np.array([])  # Will be filled during the simulation
        self.voltage_angle: np.ndarray = np.array([])  # Will be filled during the simulation
        self.voltage_angle_degrees: np.ndarray = np.array([])  # Will be filled during the simulation
        # self.logger.debug("Initiated.")

    def setup(self):
        self.lmp = np.zeros(self.simulation.time.size) # Create an empty list of LMPs
        self.voltage_angle = np.zeros(self.simulation.time.size) # Create an empty list of Voltage Angles
        self.voltage_angle_degrees = np.zeros(self.simulation.time.size) # Create an empty list of Voltage Angles

    def remove(self):
        """Remove self from the simulation"""
        self.simulation.nodes.remove(self)

    @property
    def lses_residential(self):
        return [i for i in self.lses if i.sector == "Residential"]

    @property
    def lses_commercial(self):
        return [i for i in self.lses if i.sector == "Commercial"]

    @property
    def lses_industrial(self):
        return [i for i in self.lses if i.sector == "Industrial"]
