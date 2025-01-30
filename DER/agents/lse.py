"""
lse.py
"""
import numpy as np

from DER.agents.abstract_agent import AbstractAgent
from ..adoption import let_customers_decide
from ..adoption_TVA import let_customers_decide as let_tva_customers_decide


class LSE(AbstractAgent):
    """Load Servicing Entity (LSE) Agent"""
    def __init__(
        self,
        node,
        initial_number_of_customers: int = None,  # Initial Number of Customers
        average_hourly_demand_per_customer: float = 0.001283,  # in MW/h
        hourly_demands_per_customer: list = None,
        name: str = None,
        # Optional paramters for testing and calibration
        customers_can_detach: bool = True,
        verbose: bool = True,
        sector="Residential",
        fixed_demand: float = None,
    ):
        """
        Load Servicing Entity (LSE).
        using a an average hourly demand of 0.001283 MW/h, which is the mena residential consumption
        """
        super().__init__()
        # assign the node and the simulation to the object
        self.node = node
        self.node.lses.append(self)
        self.simulation = self.node.simulation
        self.simulation.lses.append(self)
        # Variables
        self.name = name if name else f"LSE {self.simulation.lses.index(self) + 1}"
        
        self.use_fixed_demand = False
        self.use_hourly_demands_per_customer = False
        self.use_average_hourly_demand_per_customer = False
        if fixed_demand is not None:
            self.fixed_demand = fixed_demand
            self.use_fixed_demand = True
            assert hourly_demands_per_customer is None, "A fixed demand is already given"
            assert initial_number_of_customers is None
            assert customers_can_detach is None
            # print(f"LSE: {self.name} is using fixed demand.")
        elif hourly_demands_per_customer is not None:  # used in TVA
            self.hourly_demands_per_customer = hourly_demands_per_customer
            self.use_hourly_demands_per_customer = True
            assert len(hourly_demands_per_customer) == 24, "Must be values for 24 hours"
            # print(f"LSE: {self.name} is using hourly demands per customer.")
        else:
            self.average_hourly_demand_per_customer = average_hourly_demand_per_customer
            self.use_average_hourly_demand_per_customer = True
            # print(f"LSE: {self.name} is using an average hourly demand per customer.")

        self.customers_can_detach = customers_can_detach
        self.initial_number_of_customers = initial_number_of_customers

        self.verbose = verbose
        assert sector in ["Residential", "Commercial", "Industrial"]
        self.sector = sector
        # PV System Variables
        self.adoption_sigma = 0.2  # Variance for adoption logarithmic curve. Check adoption.py for more info.
        self.sun_hours_per_day = 4 # Check adoption.py for more info.
        # Financial variables
        self.yearly_interest = 0.06
        self.dg_investment_discount = 0.00
        self.dg_pv_only_investment_discount = 0.00

    def __str__(self):
        return f"LSE: '{self.name}' @'{self.node.name}'"

    @property
    def lmp(self):
        return self.node.lmp
    
    def setup_one_time(self):
        self.demand = (
            self.initial_number_of_customers
            * self.average_hourly_demand_per_customer
        )

    def setup(self):
        self.demand = np.zeros(self.simulation.time.size)
        self.demand[:] = np.nan
        self.number_of_active_customers = np.zeros(self.simulation.time.size)
        self.number_of_active_customers[:] = np.nan
        self.number_of_customers_leaving = np.zeros(self.simulation.time.size)
        self.number_of_customers_leaving[:] = np.nan

    def remove(self):
        # Remove self from the simulation
        self.simulation.lses.remove(self)
        self.node.lses.remove(self)
    
    def _tick_onetime_simulation(self):
        self.demand[self.simulation.tick_counter] = self.fixed_demand[
            self.simulation.tick_time.time().hour
        ]

    def tick_before_solver(self):
        """Put everything that happens during a tick before the solver here"""
        if not self.is_detached:
            self.number_of_customers_leaving[
                self.simulation.tick_counter
            ] = 0  # just to set it

            if self.simulation.tick_counter == 0:
                self.number_of_active_customers[
                    self.simulation.tick_counter
                ] = self.initial_number_of_customers
            else:
                self.number_of_active_customers[self.simulation.tick_counter] = (
                    self.number_of_active_customers[
                        self.simulation.tick_counter - 1
                    ]
                    - self.number_of_customers_leaving[
                        self.simulation.tick_counter - 1
                    ]
                )

            if self.number_of_active_customers[self.simulation.tick_counter] <= 1:
                self.is_detached = True
            else:
                if self.use_average_hourly_demand_per_customer:
                    self.demand[self.simulation.tick_counter] = (
                        self.number_of_active_customers[
                            self.simulation.tick_counter
                        ]
                        * self.average_hourly_demand_per_customer
                        * hour_factor(hour=self.simulation.tick_time.time().hour)
                    )
                elif self.use_hourly_demands_per_customer:
                    self.demand[self.simulation.tick_counter] = (
                        self.number_of_active_customers[
                            self.simulation.tick_counter
                        ]
                        * self.hourly_demands_per_customer[
                            self.simulation.tick_time.time().hour
                        ]
                    )

    def tick_after_solver(self):
        if self.customers_can_detach:
            if self.node.simulation.use_tva_adoption_equations:
                let_tva_customers_decide(self)
            else:
                let_customers_decide(self)


def hour_factor(hour: int) -> float:
    """
    Returns a factor for the hour.
    :param hour: int between 0 and 23
    :return: factor as float
    """
    c_a = 0.24038391
    c_b = 0.26130573
    c_c = 1.74283148
    c_d = 0.99998385
    return c_a * np.cos(hour * c_b + c_c) + c_d
