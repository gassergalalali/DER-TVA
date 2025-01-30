"""
generator.py
"""
import numpy as np
import scipy.stats

from DER.agents.abstract_agent import AbstractAgent
from DER.agents.node import Node

from ..logger import get_logger

class Generator(AbstractAgent):
    def __init__(
        self,
        node: Node,
        a: float,  # Generator Cost function parameter a
        b: float,  # and b
        fixed_cost: float,  # and fixed generator cost
        p_max: float,  # Max capacity of generators
        p_min: float = 0,  # Minimum capacity for generators
        ramp_rate: float = None,
        force_start_commitment: float = None, # Force the commitment at start at value
        emissions: float = None,
        name: str = None,
        learner=None,
        retirement_date=None,
    ):
        super().__init__()
        # assign the node and the simulation to the object
        self.node = node
        self.node.generators.append(self)
        self.simulation = self.node.simulation
        self.simulation.generators.append(self)
        # These are the inputs of the generator
        assert a > 0
        assert b > 0
        self.initial_a = a
        self.initial_b = b
        self.fixed_cost = fixed_cost
        self._hard_p_max = p_max
        self._hard_p_min = p_min
        self.ramp_rate = ramp_rate
        self.force_start_commitment = force_start_commitment
        # if the reinforced learning is enabled, it will be created in the setup method.
        self.learner = learner
        # Extra Variabls
        self.name: str = name
        self.plant = None  # Assign the generator to a plant
        self.emissions: float = (
            emissions  # The emmisions per hour will be assigned here
        )
        self.retirement_date = retirement_date
        # Setup
        self.generation: np.ndarray = np.array(
            []
        )  # To be initiated in the setup() and filled by the interface

    @property
    def lmp(self) -> np.array:
        """Returns the LMP (which is an array) from the node where the generator is"""
        return self.node.lmp

    def setup(self, force_fixed_parameters=False):
        # Initiate the generation output of the generator.
        # It will be filled later by the solver interface.
        assert not (force_fixed_parameters and self.learner)
        if force_fixed_parameters:
            self.a = self.initial_a
            self.b = self.initial_b
            self.p_max = self._hard_p_max
            self.p_min = self._hard_p_min
            self.generation = 0
        else:
            empty_array = np.zeros(self.node.simulation.time.size)
            empty_array[:] = np.nan
            self.a = np.full(self.node.simulation.time.size, self.initial_a)
            self.b = np.full(self.node.simulation.time.size, self.initial_b)
            self.generation = np.zeros(self.node.simulation.time.size)
            # Set up reinforced learning if enabled
            if self.learner:
                self.learner.setup(
                    number_of_iterations=len(
                        [a for a in self.simulation.time.tolist() if a.hour == 0]
                    )
                )
            self.variable_cost = empty_array.copy()
            self.revenue = empty_array.copy()
            self.profit = empty_array.copy()
            self.profit_percentage = empty_array.copy()
            self.p_max = empty_array.copy()
            self.p_max[0] = self._hard_p_max
            self.p_min = empty_array.copy()
            self.p_min[0] = self._hard_p_min

    def remove(self):
        """Remove self from the simulation"""
        self.simulation.generators.remove(self)
        self.node.generators.remove(self)

    def tick_before_solver(self):
        """Do a tick before the interface is run"""
        if self.simulation.tick_counter == 0:
            if self.force_start_commitment is not None:
                print(f"[generator.py] Forcing generation")
                self.p_max[0] = self.force_start_commitment + 0.001
                self.p_min[0] = self.force_start_commitment - 0.001

        if self.simulation.tick_counter > 0:
            # Update the capacities if a ramp rate is given
            if self.ramp_rate is not None:
                self.p_max[self.simulation.tick_counter] = min(
                    self._hard_p_max,
                    self.generation[self.simulation.tick_counter - 1] + self.ramp_rate,
                )
                self.p_min[self.simulation.tick_counter] = max(
                    self._hard_p_min,
                    self.generation[self.simulation.tick_counter - 1] - self.ramp_rate,
                )
            else:
                self.p_max[self.simulation.tick_counter] = self._hard_p_max
                self.p_min[self.simulation.tick_counter] = self._hard_p_min
        
        if hasattr(self, 'retirement_date'):
            if self.retirement_date:
                if self.simulation.tick_time >= self.retirement_date:
                    if not self.is_detached:
                        self.is_detached = True
                        get_logger().info(f"Generators {self.name}: Detaching on {self.simulation.tick_time}")
                        # print(f"Generators {self.name}: Detaching on {self.simulation.tick_time}")

        # Update the Learner if it is used
        if self.simulation.tick_counter > 0:
            if not self.learner:
                self.a[self.simulation.tick_counter] = self.initial_a
                self.b[self.simulation.tick_counter] = self.initial_b
            else:  # Then the Reinforced Learning is enabled
                if self.simulation.tick_time.hour == 0:
                    action = self.learner.get_action()
                    self.a[self.simulation.tick_counter] = self.initial_a * (1 + action)
                    self.b[self.simulation.tick_counter] = self.initial_b
                else:
                    self.a[self.simulation.tick_counter] = self.a[
                        self.simulation.tick_counter - 1
                    ]
                    self.b[self.simulation.tick_counter] = self.b[
                        self.simulation.tick_counter - 1
                    ]
        


    def tick_after_solver(self):
        """ Cost, Revenue, and Profit """
        """ NOTE: The Cost is according to the INITIAL paramters """
        self.variable_cost[
            self.simulation.tick_counter
        ] = self.initial_a * self.generation[
            self.simulation.tick_counter
        ] + self.initial_b * np.power(
            self.generation[self.simulation.tick_counter], 2
        )
        self.revenue[self.simulation.tick_counter] = (
            self.node.lmp[self.simulation.tick_counter]
            * self.generation[self.simulation.tick_counter]
        )
        self.profit[self.simulation.tick_counter] = (
            self.revenue[self.simulation.tick_counter]
            - self.variable_cost[self.simulation.tick_counter]
            - self.fixed_cost
        )
        self.profit_percentage[self.simulation.tick_counter] = (
            self.profit[self.simulation.tick_counter] 
            / (self.variable_cost[self.simulation.tick_counter] + self.fixed_cost)
        ) if (self.variable_cost[self.simulation.tick_counter] + self.fixed_cost) != 0 else 0
        """ Learner """
        if self.simulation.tick_counter > 0:
            if self.learner:
                if self.simulation.tick_time.hour == 0:
                    profit_day = sum(self.profit[self.simulation.tick_counter - 24: self.simulation.tick_counter])
                    cogs_day = (
                        sum(self.variable_cost[self.simulation.tick_counter - 24: self.simulation.tick_counter])
                        + 24 * self.fixed_cost
                    )
                    reward = (
                        profit_day
                        / cogs_day
                    ) if cogs_day != 0 else 0
                    self.learner.feedback_reward(reward)
