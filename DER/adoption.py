"""
adoption.py
Creates adoption of DSG in LSEs
"""
import numpy as np
import scipy.stats


from .solar_system import get_system_cost, payment_loan


def let_customers_decide(
    lse,
    ):
    """ Actions after the solver """
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
                    if not lse.is_detached:
                        # How much did a resident in this LSE pay in the last month?
                        paid_in_last_month = np.sum(
                            lse.average_hourly_demand_per_customer
                            * lse.node.lmp[
                                lse.simulation.tick_counter
                                - (30 * 24) : lse.simulation.tick_counter
                            ]
                        )
                        # How much does a resident expect to pay in the next 25 years?
                        expected_to_pay_in_25_years = paid_in_last_month * 12 * 25

                        dg_investment = get_system_cost(
                            average_hourly_demand_in_MW=lse.average_hourly_demand_per_customer,
                            sun_hours_per_day=lse.sun_hours_per_day,
                            date=lse.simulation.tick_time.date(),
                        )

                        dg_investment = dg_investment * (1 - lse.dg_investment_discount)

                        loan_payment = payment_loan(
                            dg_investment, 
                            lse.yearly_interest
                            )

                        # How many can leave?
                        # If the expected total bills to be paid in the following 25 years is lower
                        # than the investment to install solar cells
                        # sigma = 0.2
                        sigma = lse.adoption_sigma
                        percentage_disconnect = scipy.stats.lognorm.cdf(
                            x=paid_in_last_month,
                            s=sigma,
                            scale=loan_payment,
                        )

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
                        if lse.verbose:
                            print(
                                f"\n LSE: {lse}"
                                + f"\n Average Hourly Demand (Kwh/hour/customer): {lse.average_hourly_demand_per_customer * 1000:10,.2f}"
                                + f"\n Daily Demand (Kwh/Day/customer): {lse.average_hourly_demand_per_customer * 1000 *24:10,.2f}"
                                + f"\n Montly Demand (Kwh/Month/customer): {lse.average_hourly_demand_per_customer * 1000 *24 * 30:10,.2f}"
                                + f"\n Amount Paid in Last Month ($/Customer): {paid_in_last_month:10,.2f}"
                                + f"\n Cost of Next 25 Years of Power ($/Customer): {expected_to_pay_in_25_years:10,.2f}"
                                + f"\n Required Investment to Install DG ($/Customer): {dg_investment:10,.2f}"
                                + f"\n Monthly Loan Payment ($/Month): {loan_payment:10,.2f}"
                                + f"\n Average LMP ($/MW): {np.mean(lse.node.lmp[0:lse.simulation.tick_counter]):10,.2f}"
                                + f"\n Percentage to disconnect (%): {percentage_disconnect:10,.2%}"
                                + f"\n Number of Customers leaving (count): {number_of_customers_leaving:10,}"
                            )