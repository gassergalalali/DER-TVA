"""
adoption.py
Another function for adtoption of DG
from Dr. Charles and Tim
〖Pr⁡[adopt]〗_it=β_1i 〖bill〗_it+β_2i 〖emissions〗_t+β_3i 〖PVcost〗_t
Uses Ramsey Pricing
"""
import numpy as np
import rich

from .solar_system import get_system_cost, payment_loan
from .logger import get_logger


def let_customers_decide(lse):
    """ Actions after the solver """
    logger = get_logger()
    # Check if first year has passed
    if (
        lse.simulation.tick_time - lse.simulation.start_time
    ).days > 30:  # pass the first month
        # Check if this is the first day on a month
        if lse.simulation.tick_time.date() == lse.simulation.tick_time.date().replace(
            day=1
        ):
            # Check if it is the start of a day
            if lse.simulation.tick_time.time().hour == 0:
                if lse.customers_can_detach:
                    if lse.ramsey_price_was_calculated:
                        if not lse.is_detached:
                            # The Percentage adoption has THREE parts
                            # Part 1: Savings on electricity bill
                            coef_save = lse.coef_save  # From the Excel
                            ramsey_price = lse.ramsey_price[lse.simulation.tick_counter]
                            assert not np.isnan(ramsey_price), "Price shouldn't be nan."
                            savings_from_DSG_system = lse.node.savings_from_DSG_system
                            part1 = (
                                coef_save 
                                * ramsey_price 
                                * savings_from_DSG_system / 1000
                                # Note: savings_from_DSG_system is converted from kwh/month to Mwh/month
                            )
                            # Part 2: Emissions
                            coef_em = lse.coef_em
                            emmisions_at_node = sum(  # E_it -> the amount of electricity generated by TVA at node i
                                [
                                    sum(
                                        g.emissions  # The Emmisions is in Tons/MWh
                                        * g.generation[
                                            lse.simulation.tick_counter
                                            - 30 * 24 : lse.simulation.tick_counter
                                        ]
                                    )
                                    for g in lse.node.generators
                                ]
                            )
                            generation_at_node = sum(  # q_it -> the amount of electricity generated by TVA at node i
                                [
                                    sum(
                                        g.generation[
                                            lse.simulation.tick_counter
                                            - 30 * 24 : lse.simulation.tick_counter
                                        ]
                                    )
                                    for g in lse.node.generators
                                ]
                            )

                            customer_electricity_usage = (
                                sum([i.demand[lse.simulation.tick_counter] for i in lse.node.lses])
                                / sum([i.number_of_active_customers[lse.simulation.tick_counter] for i in lse.node.lses])
                            )

                            assert not np.isnan(emmisions_at_node), emmisions_at_node
                            assert not np.isnan(generation_at_node), generation_at_node
                            if generation_at_node > 0:
                                part2 = (
                                    coef_em  # Coefficients from Scott
                                    * emmisions_at_node  # sum(emision * generation) for each node
                                    / generation_at_node  # sum(generation) at each node
                                    * (
                                        customer_electricity_usage   # (sum(demand) / #customers) for each node
                                        - (savings_from_DSG_system / 1000)  # From Table 1 and divided by 1000 to convert from KW to MW
                                    ) 
                                    / customer_electricity_usage  # (sum(demand) / #customers) for each node
                                )
                            else:
                                logger.debug(f"[adoption_TVA.py] Generation at lse {lse.name} is zero.")
                                part2 = 0
                            # Part 3: PV Cost
                            coef_cost = lse.coef_cost
                            total_pv_system_cost = get_system_cost(  # includes system and batteries
                                savings_from_DSG_system
                                / 30
                                / 24
                                / 1000,  # Savings in kwh/month converted to MW/h,
                                date=lse.simulation.tick_time.date(),
                                sun_hours_per_day=lse.sun_hours_per_day,  # Adding the sun-hours from the LSE
                                system_discount = lse.dg_investment_discount,
                                pv_only_discount = lse.dg_pv_only_investment_discount,
                            )
                            total_pv_system_cost = total_pv_system_cost * (1 - lse.dg_investment_discount)  # Rebate
                            monthly_pv_system_cost = payment_loan(total_pv_system_cost)
                            part3 = coef_cost * monthly_pv_system_cost
                            # Calculate prob
                            const = lse.coef_const
                            log_odds = (part1 + part2 + part3 + const)
                            odds = np.exp(log_odds)
                            percentage_disconnect = odds / (odds + 1) / 240  
                            # Record the number of customers leaving
                            number_of_customers_leaving = np.round(
                                percentage_disconnect
                                * lse.number_of_active_customers[
                                    lse.simulation.tick_counter
                                ],
                                0,
                            )

                            lse.number_of_customers_leaving[
                                lse.simulation.tick_counter
                            ] = number_of_customers_leaving

                            # debug the numbers
                            message = (
                                f"[adoption_TVA.py]:"
                                + f"\n LSE: {lse} @ Iteration {lse.simulation.tick_counter}: Time: {lse.simulation.tick_time.date()}"
                                + f"\n Ramsey ($/MW): {ramsey_price:10,.2f}"
                                + f"\n Part1 (Savings)"
                                + f"\n     = coef_save[{coef_save}] * ramsey_price[{ramsey_price:,.2f}] * savings_from_DSG_system[{savings_from_DSG_system:,.2f}] / 1000"
                                + f"\n     = {part1:10,.2f}"
                                + f"\n Part2 (Emmissions)"
                                + f"\n     = coef_em[{coef_em}]"
                                + f"\n       * emmisions_at_node[{emmisions_at_node:,.2f}]"
                                + f"\n       / generation_at_node[{generation_at_node:,.2f}]"
                                + f"\n       * ("
                                + f"\n            customer_electricity_usage[{customer_electricity_usage:,.2f}]"
                                + f"\n            - (savings_from_DSG_system[{savings_from_DSG_system:,.2f}] / 1000)"
                                + f"\n         )"
                                + f"\n       / customer_electricity_usage[{customer_electricity_usage:,.2f}]"
                                + f"\n     = {part2:10,.2f}"
                                + f"\n Part3 (PV Cost)"
                                + f"\n     = coef_cost[{coef_cost}] * monthly_pv_system_cost[{monthly_pv_system_cost:,.2f}]"
                                + f"\n     = {part3:10,.2f}"
                                + f"\n Constant = {const:10,.2f}"
                                + f"\n log_odds"
                                + f"\n     = Part1 + Part 2 + Part 3 + Const"
                                + f"\n     = {part1:,.2f} + {part2:,.2f} + {part3:,.2f} + {const:,.2f}"
                                + f"\n     = {log_odds}"
                                + f"\n log_odds = Part1 + Part 2 + Part 3 + Const = {part1:,.2f} + {part2:,.2f} + {part3:,.2f} + {const:,.2f} = {log_odds:,.2f}"
                                + f"\n odds = exp(log_odds) = exp({log_odds:,.2f}) = {odds:,.2f}"
                                + f"\n Percentage to disconnect (%)"
                                + f"\n     = odds[{odds:,.2f}] / (odds[{odds:,.2f}] + 1) / 120"
                                + f"\n     = {percentage_disconnect}"
                                + f"\n Number of Customers leaving (count)"
                                + f"\n     = percentage_disconnect[{percentage_disconnect:,.2f}] * number_of_active Customers[{lse.number_of_active_customers[lse.simulation.tick_counter]}]"
                                + f"\n     =~ {number_of_customers_leaving:f} customers"
                            )
                            logger.debug(message)
                            assert percentage_disconnect >= 0, f"Percentage to disconnect is {percentage_disconnect:,.2%} -> {number_of_customers_leaving:,.2f} customers!"
                            assert percentage_disconnect <= 1, f"Percentage to disconnect is {percentage_disconnect:,.2%} -> {number_of_customers_leaving:,.2f} customers!"
