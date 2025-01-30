"""
simulation.py
This is the simulation package. It is the main object that controls everything.
"""
import datetime
import typing

import numpy as np
import tqdm
import rich

from DER.agents.generator import Generator
from DER.agents.lse import LSE
from DER.agents.plant import Plant
from DER.agents.line import Line
from DER.interface import solve

from .ramsey_pricing import calculate_ramsey_prices_tick
from .ramsey_pricing import setup as ramsey_setup

from .logger import get_logger

class Simulation:
    def __init__(
            self,
            start_datetime: datetime.datetime,
            finish_datetime: datetime.datetime,
            verbose: bool = False,
            calculate_ramsey_prices: bool = False,
            calculate_ramsey_prices_once: bool = False,
            calculate_ramsey_prices_24_hours: bool = False,
            use_tva_adoption_equations: bool = False,
    ):
        """
        The Simulation class.

        :param start_datetime:
        :param finish_datetime:

        :return: Simulation()
        """
        logger = get_logger()
        logger.debug("[simulation.py] __init__ starting.")
        assert isinstance(start_datetime, datetime.datetime)
        assert isinstance(finish_datetime, datetime.datetime)
        self.start_time = start_datetime
        self.finish_time = finish_datetime
        self.verbose = verbose
        # The agents in the simulation
        self.nodes = []
        self.lines: typing.List[Line] = []
        self.generators: typing.List[Generator] = []
        self.plants: typing.List[Plant] = []
        self.lses: typing.List[LSE] = []
        # Variables that control the simulation
        self.tick_counter: int = 0  # the tick counter starts at zero when the simulation starts
        self.tick_time: datetime.datetime = self.start_time
        # Setup
        self.time: np.ndarray = np.array([])  # Will contain all the timeline
        # Blanks
        self.tick_counter:int = None
        # TVA
        self.calculate_ramsey_prices = calculate_ramsey_prices
        self.calculate_ramsey_prices_once = calculate_ramsey_prices_once
        self.use_tva_adoption_equations = use_tva_adoption_equations
        self.calculate_ramsey_prices_24_hours = calculate_ramsey_prices_24_hours
    
    def set_start_time(self, start_datetime: datetime.datetime = datetime.datetime.today()):
        self.start_time = datetime.datetime.fromisoformat(start_datetime.date().isoformat())
        self.tick_time: datetime.datetime = self.start_time
    
    def set_finish_datetime(self, finish_datetime: datetime.datetime = datetime.datetime.today()):
        self.finish_time = datetime.datetime.fromisoformat(finish_datetime.date().isoformat())

    def setup(self):
        """Setup the Simulation for a run"""
        # Set the time range array
        self.time = np.arange(
            self.start_time,
            self.finish_time,
            datetime.timedelta(hours=1)
        )
        # Reset the tick counter and tick time
        self.tick_counter = 0
        self.tick_time = self.start_time
        for node in self.nodes:
            node.setup()
        for line in self.lines:
            line.setup()
        for generator in self.generators:
            generator.setup()
        for lse in self.lses:
            lse.setup()
        if self.calculate_ramsey_prices:
            ramsey_setup(self)

    def run(self, show_progress:bool = True):
        """Start the simulation"""
        # Tick Tok...
        if show_progress: 
            iterations = tqdm.tqdm(range(len(self.time)), desc="")
        else:
            iterations = range(len(self.time))
        for tick_number in iterations:  # tqdm is used to show a progress bar...
            # Increase the tick counter
            self.tick_counter = tick_number  # set the tick counter so that every agent can use it
            self.tick_time = self.start_time + datetime.timedelta(hours=self.tick_counter)
            # iterations.set_description("Simulation @" + str(self.tick_time))
            # Start the tick operations
            """ Do a tick before solver for each LSE """
            for node in self.nodes:
                for lse in node.lses:
                    lse.tick_before_solver()
            """ Fo a tick for each Generator too """
            for generator in self.generators:
                generator.tick_before_solver()
            """ Do a tick for each line """
            for line in self.lines:
                line.tick_before_solver()
            """ Pre-solve Check """
            if all([lse.is_detached for lse in self.lses]):
                print("[Simulation] All Lses are detached. Breaking.")
                break
            for node in self.nodes:
                total_min_generation = sum([g.p_min[self.tick_counter] for g in node.generators])
                total_max_generation = sum([g.p_max[self.tick_counter] for g in node.generators])
                total_demands = sum([lse.demand[self.tick_counter] for lse in node.lses])
                total_transmission = sum([line.max_capacity for line in node.lines])
                if total_demands + total_transmission < total_min_generation:
                    rich.print(
                        f"[simulation.py:WARNING] {node.name}: Demand + Transmission Capacity < Minimum Generation"
                        f"\n    Total demand = {total_demands:,.2f}"
                        f"\n    Transmission Capacity = {total_transmission:,.2f}"
                        f"\n    Minimum Generation {total_min_generation:,.2f}"
                        f"\n    Maximum Generation {total_max_generation:,.2f}"
                    )
                    raise Exception
                if total_demands > total_max_generation + total_transmission:
                    rich.print(
                        f"[simulation.py:WARNING] {node.name}: Demand > Maximum Generation + Transmission"
                        f"\n    Total demand = {total_demands:,.2f}"
                        f"\n    Maximum Generation {total_max_generation:,.2f}"
                        f"\n    Transmission Capacity = {total_transmission:,.2f}"
                        f"\n    Minimum Generation {total_min_generation:,.2f}"
                    )
                    raise Exception
            """ Run the simulation """
            # The Interface encodes the simulation, runs it, and stores the results.
            solve(simulation=self)
            """ Calcualte the Ramsey Prices if set to do that """
            if self.calculate_ramsey_prices:
                calculate_ramsey_prices_tick(self)
            """ Tick the Investment Ruler """
            for lse in self.lses:
                lse.tick_after_solver()
            """ Tick the generators """
            for generator in self.generators:
                generator.tick_after_solver()
