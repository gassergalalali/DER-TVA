from contextlib import ExitStack
import copy
import numpy as np
import scipy.optimize

from .logger import get_logger


def setup(simulation):
    simulation.ramsey_memory_last_tick_counter = None
    simulation.ramsey_memory_calculated_before = False

    for lse in simulation.lses:
        lse.ramsey_price = np.zeros(simulation.time.size)
        lse.ramsey_price[:] = np.nan
        lse.ramsey_price_was_calculated = False

    # Do some checks
    for node in simulation.nodes:
        for residential_lse in node.lses_residential:
            assert residential_lse.c == node.lses_residential[0].c
            assert residential_lse.d == node.lses_residential[0].d

def calculate_ramsey_prices_tick(simulation):
    """
    Special Function to calculate Ramsey Prices.
    Done at six-months interaval
    """
    logger = get_logger()
    if simulation.ramsey_memory_last_tick_counter is None:
        simulation.ramsey_memory_last_tick_counter = simulation.tick_counter

    if simulation.tick_time.day == 1:
        if simulation.tick_time.hour == 0:
            if simulation.tick_time.month in [1, 6]:
                if simulation.tick_counter > 0:
                    if simulation.ramsey_memory_calculated_before and simulation.calculate_ramsey_prices_once:
                        logger.warning(
                            "[ramsey_pricing.py] Not Calculating Ramsey Prices because calculate once"
                        )
                    else: 
                        simulation.ramsey_memory_calculated_before = True
                        logger.info("[ramsey_pricing.py] Calculating Ramsey Prices.")
                        a = simulation.ramsey_memory_last_tick_counter
                        b = simulation.tick_counter
                        if simulation.calculate_ramsey_prices_24_hours:
                            # TODO: make this for 24 hours and weighted
                            total_network_cost = sum(
                                [
                                    sum(
                                        g.a[b-24:b] * g.generation[b-24:b]
                                        + g.b[b-24:b] * (g.generation[b-24:b] ** 2)
                                    )
                                    for g in simulation.generators
                                ]
                            )
                            total_network_power = sum(
                                [sum(g.generation[b-24:b]) for g in simulation.generators]
                            )
                        else:
                            total_network_cost = sum(
                                [
                                    sum(
                                        g.a[a:b] * g.generation[a:b]
                                        + g.b[a:b] * (g.generation[a:b] ** 2)
                                    )
                                    for g in simulation.generators
                                ]
                            )
                            total_network_power = sum(
                                [sum(g.generation[a:b]) for g in simulation.generators]
                            )
                        AVC = total_network_cost / total_network_power
                        ramsey_optimization(simulation, AVC)

    for lse in simulation.lses:
        # Catch all NANs and set them as the previous Ramsey Prices if available
        if np.isnan(lse.ramsey_price[simulation.tick_counter]):
            if not np.isnan(lse.ramsey_price[simulation.tick_counter - 1]):
                lse.ramsey_price[simulation.tick_counter] = lse.ramsey_price[
                    simulation.tick_counter - 1
                ]


def _calculate_ramsey_for_node(ramsey_price_residential, node, AVC):
    t = node.simulation.tick_counter
    # Record the new Ramsey Prices to the Residential Customers
    for residential_lse in node.lses_residential:
        residential_lse.ramsey_price[t] = ramsey_price_residential[0]
        assert residential_lse.ramsey_price[t] > 0, residential_lse.ramsey_price[t]
    # Create temporary variable to calculate profits
    profits = []
    profits.append(-node.tva_implied_fixed_cost)
    # Calculate the Profit for Residential LSEs
    for residential_lse in node.lses_residential:
        residential_lse.tva_profit = (
            (residential_lse.ramsey_price[t] - AVC)
            * residential_lse.tva_number_of_customers
            * np.exp(residential_lse.c - residential_lse.d * np.log(residential_lse.ramsey_price[t]))
        )
        profits.append(residential_lse.tva_profit)
    # Calculate the Price for the Commercial LSEs
    assert len(node.lses_commercial) == 1
    commercial_lse = node.lses_commercial[0]
    commercial_lse.ramsey_price[t] = AVC / (
        1
        - (residential_lse.d / commercial_lse.d)
        * (1 - AVC / residential_lse.ramsey_price[t])
    )
    assert commercial_lse.ramsey_price[t] > 0, f"{commercial_lse.ramsey_price[t]=}"
    # Calculate the Profit for the Commercial LSEs
    commercial_lse.tva_profit = (
        (commercial_lse.ramsey_price[t] - AVC)
        * commercial_lse.tva_number_of_customers
        * np.exp(commercial_lse.c - commercial_lse.d * np.log(commercial_lse.ramsey_price[t]))
    )
    profits.append(commercial_lse.tva_profit)
    # Calculate the Price for the Industrial LSEs (if any)
    assert len(node.lses_industrial) <= 1
    if node.lses_industrial != []:
        industrial_lse = node.lses_industrial[0]
        industrial_lse.ramsey_price[t] = AVC / (
            1
            - (residential_lse.d / industrial_lse.d)
            * (1 - AVC / residential_lse.ramsey_price[t])
        )
        assert industrial_lse.ramsey_price[t] > 0, industrial_lse.ramsey_price[t]
        # Calculate the Profit for the Industrial LSEs
        industrial_lse.tva_profit = (
            (industrial_lse.ramsey_price[t] - AVC)
            * industrial_lse.tva_number_of_customers
            * np.exp(industrial_lse.c - industrial_lse.d * np.log(industrial_lse.ramsey_price[t]))
        )
        profits.append(industrial_lse.tva_profit)
    # Calculate the total node Profit
    node.tva_profit = np.sum(profits)
    return pow(node.tva_profit, 2)


def ramsey_optimization(simulation, AVC):
    logger = get_logger()
    for node in simulation.nodes:
        res = scipy.optimize.minimize(
            fun=lambda x: _calculate_ramsey_for_node(x, node, AVC),
            x0=[
                AVC - 1,
            ],
            bounds=((1, 200),),
            method="Powell",
        )
        logger.debug(
            f"[ramsey_pricing.py] Ramsey Pricing Optimization: {node.name:20} : {res['message']} : x = {res['x']}"
        )
        for lse in node.lses:
            lse.ramsey_price_was_calculated = True
