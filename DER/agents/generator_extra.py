"""
generator_extra.py
This generator will only have capacity if the grid's capacity is less than demand.
"""
import numpy as np

from DER.agents.generator import Generator
from DER.agents.node import Node

from ..logger import get_logger

class GeneratorExtra(Generator):
    def __init__(
        self,
        node: Node,
        a: float,
        b: float,
        fixed_cost: float,
        p_max: float,
        p_min: float = 0,
        ramp_rate: float = None,
        force_start_commitment: float = None,
        emissions: float = None,
        name: str = None,
        learner=None,
        retirement_date=None,
    ):
        super().__init__(
            node,
            a,
            b,
            fixed_cost,
            p_max,
            p_min,
            ramp_rate,
            force_start_commitment,
            emissions,
            name,
            learner,
            retirement_date,
        )

    def tick_before_solver(self):
        super().tick_before_solver()
        self.p_max[self.simulation.tick_counter] = 0
        network_total_demand = np.sum(
            [lse.demand[self.simulation.tick_counter] for lse in self.simulation.lses]
        )
        network_total_generation_capacity = np.sum(
            [g.p_max[self.simulation.tick_counter] for g in self.simulation.generators if not g.is_detached]
        )
        network_deficit = network_total_demand - network_total_generation_capacity
        if network_deficit > 0: # The max cap is only as big as the deficit in the network
            get_logger().info(f"[Warning!][generator_extra] Total Network Deficit = {network_total_demand:.2f} - {network_total_generation_capacity:.2f} = {network_deficit:.2f}")
            # print(f"[Warning!][generator_extra] Total Network Deficit = {network_total_demand:.2f} - {network_total_generation:.2f} = {network_deficit:.2f}")
            assert network_deficit <= self._hard_p_max, f"The generator does not have enough capacity to cover the deficit. {network_deficit=:.2f} {self._hard_p_max=:.2f}"
            self.p_max[self.simulation.tick_counter] = network_deficit + 0.01
            get_logger().info(f"[generator_extra] p_max adjusted to {self.p_max[self.simulation.tick_counter]:.2f}")
            # print(f"[generator_extra] p_max adjusted to {self.p_max[self.simulation.tick_counter]}")
        
