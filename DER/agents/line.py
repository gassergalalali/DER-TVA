"""
line.py

Transmission lines connect Nodes.
A Transmission line connects two nodes.
Each node is of the "Node" class.
Each transmission line has:
 - Reactance (Ohms)
 - Maximum Capacity (MWs)
"""
import numpy as np

from DER.agents.abstract_agent import AbstractAgent
from DER.agents.node import Node


class Line(AbstractAgent):
    def __init__(
            self,
            node1: Node = None,
            node2: Node = None,
            reactance=0.1,
            max_capacity=100,
    ):
        """
        A Transmission line that connects two nodes.
        :param node1: Node, must be of type Node
        :param node2: The other Node being connected to.
        :param reactance: The reactance of the line in Ohms.
        :param max_capacity: The maximum capacity of the line in MWs.
        """
        super().__init__()
        # Check the Nodes are is provided
        if not node1:
            raise Exception("The node1 is missing!")
        if not node2:
            raise Exception("The node2 is missing!")

        if node1.simulation != node2.simulation:
            raise Exception("node1 and node2 are not in the same simulation!")
        # inputs
        self.node1 = node1
        self.node1.lines.append(self)  # Append the current line to the node's list of lines.
        self.node2 = node2
        self.node2.lines.append(self)  # Also append the current line to the other node's list of lines
        self.simulation = self.node1.simulation
        self.simulation.lines.append(self)
        # Variables
        self.reactance = reactance
        self.max_capacity = max_capacity
        self.flow = None  # To be calculated by the solver
        # Experiment
        self.yearly_increasing_cap = False
        self.yearly_increasing_cap_amount = 1.05
        self.capacity_history = {}


    def setup(self):
        # The flow will be calculated and assigned to this array
        self.flow = np.zeros(self.simulation.time.size)

    def remove(self):
        """Remove self from the simulation"""
        self.simulation.lines.remove(self)
        self.node1.lines.remove(self)
        self.node2.lines.remove(self)

    def tick_before_solver(self):
        if self.yearly_increasing_cap:
            if self.simulation.tick_counter > 0:
                if self.simulation.tick_time.year == 0:
                    if self.simulation.tick_time.month == 0:
                        if self.simulation.tick_time.hour == 0:
                            self.max_capacity = self.max_capacity * self.yearly_increasing_cap_amount
                            self.capacity_history[self.simulation.tick_time] == self.max_capacity
                            print(f"{__self__} increased capacity to {self.max_capacity}")
