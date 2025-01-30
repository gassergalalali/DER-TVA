"""
solar_system.py
Utility Functions for the LSEs to calculate the price and financing of solar systems
"""
import datetime
import numpy as np


def get_system_cost(
    average_hourly_demand_in_MW: float,
    date: datetime.datetime,
    sun_hours_per_day: float = 4,
    verbose = False,
    system_discount:float = 0.0,
    pv_only_discount:float = 0.0,
    ):
    assert isinstance(date, datetime.datetime) or isinstance(date, datetime.date), date.__class__
    assert sun_hours_per_day > 0
    assert sun_hours_per_day < 12
    # Cost of Solar panels to cover the expected required power for one dat
    # Assuming 5 hours and 80% efficiency
    pv_size = ( # in MW
        average_hourly_demand_in_MW
        * 24
        / sun_hours_per_day
        / 0.8
        * 1000000 
    )
    battery_size = ( # in MW
        average_hourly_demand_in_MW
        * 24
        / 0.75
        * 1000000 
    )

    price_of_pv_cells_per_unit = get_price_of_solar(date)
    price_of_batteries_per_unit = get_price_of_batteries(date)

    price_of_pv_cells = pv_size * price_of_pv_cells_per_unit
    price_of_pv_cells = price_of_pv_cells * (1 - pv_only_discount)  # Rebate

    price_of_batteries = 2.5 * battery_size * price_of_batteries_per_unit

    dg_investment = (
        price_of_pv_cells
        + price_of_batteries
    )
    dg_investment = dg_investment * (1 - system_discount)  # Rebate

    if verbose:
        print(f"{pv_size=}")
        print(f"{battery_size=}")
        print(f"{price_of_pv_cells_per_unit=}")
        print(f"{price_of_batteries_per_unit=}")
        print(f"Price of PV = {pv_size * price_of_pv_cells_per_unit=}")
        print(f"Price of Batteries = {2.5 * battery_size * price_of_batteries_per_unit=}")
        print(f"Price of Batteries covers = {(2.5 * battery_size * price_of_batteries_per_unit) / (dg_investment) * 100 =:,.2f} % of the System Price")
        print(f"{dg_investment=}")
    
    return dg_investment

def payment_loan(
    pv, 
    yearly_interest:float = 0.06
    ):
    # Amortized Loan
    r = yearly_interest / 12
    years = 25
    nper = years * 12
    monthly_payment = pv * r * pow(1 + r, nper) / (pow(1 + r, nper) - 1)
    assert monthly_payment > 0
    return monthly_payment


def get_price_of_solar(date: datetime.datetime) -> float:
    """
    Returns the predicted Price of Solar at a year
    Refer to the Jupyter Notebook Price of Solar.ipynb for model fitting
    Notebooks/Price of Solar.ipynb
    """
    year = date.year + (date.month - 1) / 12
    a = 1.84658771e02
    b = -9.10763091e-02
    c = 1.26943392e00
    x = year
    price_of_solar = np.exp(a + b * x) + c
    return price_of_solar


def get_price_of_batteries(date: datetime.datetime) -> float:
    """
    Returns the predicted Price of Solar at a year
    Refer to the Jupyter Notebook Price of Solar.ipynb for model fitting
    Notebooks/Price of Batteries.ipynb
    """
    year = date.year + (date.month - 1) / 12
    a, b, c = [4.59937113e02, -2.28766845e-01, 3.20808498e-02]
    x = year
    return np.exp(a + b * x) + c

if __name__ == "__main__":
    get_system_cost(
        average_hourly_demand_in_MW=0.01,
        date=datetime.datetime(2022, 1, 1),
        sun_hours_per_day= 4,
        verbose = True,
    )