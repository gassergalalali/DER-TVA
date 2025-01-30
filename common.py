import datetime
import os
import numpy as np
import pandas as pd

import DER


START_DATE = datetime.datetime(2020, 1, 1)
FINISH_DATE = datetime.datetime(2051, 1, 1)


def create_simulation(
    start_datetime: datetime.datetime, finish_datetime: datetime.datetime
):
    simulation = DER.Simulation(
        start_datetime=start_datetime,
        finish_datetime=finish_datetime,
        calculate_ramsey_prices=True,
        use_tva_adoption_equations=True
    )
    # Read the Excel File containing the TVA data
    file_path = os.path.join(os.path.dirname(__file__), "TVA.xlsx")
    print(f"Pandas is reading '{file_path}' ...")

    class Empty:
        pass

    dts = Empty()
    dts.dt_nodes = pd.read_excel(file_path, sheet_name="Nodes")
    dts.dt_lines = pd.read_excel(file_path, sheet_name="Lines")
    dts.dt_generators = pd.read_excel(file_path, sheet_name="Generators")
    dts.dt_plants = pd.read_excel(file_path, sheet_name="Plants")
    dts.dt_lses = pd.read_excel(file_path, sheet_name="LSEs")

    # Add Nodes
    for _, row in dts.dt_nodes.iterrows():
        node = DER.Node(simulation=simulation, name=str(row["Name"]))
        node.tva_implied_fixed_cost = float(row["Implied Fixed Cost"])
        # Add the Savings and Coefficcients from Dr Charles
        node.savings_from_DSG_system = float(row["Savings From DSG System"])

    # Add Lines
    for _, row in dts.dt_lines.iterrows():
        DER.Line(
            node1=simulation.nodes[int(row["from"]) - 1],
            node2=simulation.nodes[int(row["to"]) - 1],
            max_capacity=float(row["Cap"]),
            reactance=float(row["Reactance"]),
        )

    # Add Generators
    for _, row in dts.dt_generators.iterrows():
        assert float(row["a"]) > 0
        assert float(row["b"]) > 0

        generator = DER.Generator(
            name=str(row["Name"]),
            node=simulation.nodes[int(row["Node"]) - 1],
            a=float(row["a"]),
            b=float(row["b"]),
            fixed_cost=float(row["FixedCost"]),
            p_min=float(row["Min"]),
            p_max=float(row["Max"]),
            emissions=float(row["CO2 Emissions"])
            if str(row["CO2 Emissions"]) != "Missing"
            else None,
            retirement_date=datetime.datetime.fromisoformat(str(row["Retire Date"]))
            if str(row["Retire Date"]) != "nan"
            else None,
        )
        generator.plant_name = str(row["Plant Name"])
        generator.type = str(row["Type"])

    # Add Plants
    for _, row in dts.dt_plants.iterrows():
        plant_name = str(row["Name"])
        plant = DER.Plant(name=plant_name, simulation=simulation)
        plant.expected_daily_generation = float(row["Expected Daily Generation"])
        plant.location = str(row["Location"])
        for generator in simulation.generators:
            if generator.plant_name == plant_name:
                plant.attach_generator(generator)

    # Add LSEs
    for _, row in dts.dt_lses.iterrows():
        assert row["Sector"] in ["Residential", "Commercial", "Industrial"]
        assert float(row["d"]) > 0
        lse = DER.LSE(
            node=simulation.nodes[int(row["Node"]) - 1],
            name=str(row["Name"]),
            # fixed_demand = np.array([float(row[f"Fixed Demand Hour {i+1}"]) for i in range(24)]),
            hourly_demands_per_customer=np.array(
                [float(row[f"Fixed Demand Hour {i+1}"]) for i in range(24)]
            )
            / int(row["# Customers"]),
            initial_number_of_customers=int(row["# Customers"]),
            sector=row["Sector"],
        )
        lse.c = float(row["c"])
        lse.d = float(row["d"])
        lse.tva_number_of_customers = int(row["# Customers"])

        lse.coef_cost = float(row["coef_cost"])
        lse.coef_save = float(row["coef_save"])
        lse.coef_em = float(row["coef_em"])
        lse.coef_const = float(row["coef_const"])

    for lse in simulation.lses:
        lse.use_tva_adoption_equations = True  # Use the Equations from Dr. Charles

    """ Add the extra Generators """
    for node in simulation.nodes:
        if "chattan" in node.name.lower():
            # print(f"[TVA] Adding extra generator at {node.name}")
            generator_extra = DER.GeneratorExtra(
                name="Extra Generator (Hydro)",
                node=node,
                a=3.75,
                b=0.0000958026,
                fixed_cost=0,
                p_min=0,
                p_max=2000,
                emissions=0,
                retirement_date=None,
            )
            generator_extra.plant_name = None
            generator_extra.type = "Hydro (Extra)"

    assert generator_extra in simulation.generators

    return simulation
