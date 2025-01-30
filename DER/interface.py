"""
interface.py
Connects between the simulation and solver.py
"""
import numpy as np

import DER.solver

# for PU form
base_s = 100
base_v = 10


def solve(simulation, verbose=False):
    """ Encode Generators """
    a = []
    b = []
    fixed_cost = []
    p_max = []
    p_min = []
    generator_locations = []
    for generator in simulation.generators:
        if not generator.is_detached:
            assert generator.a[simulation.tick_counter] > 0
            a.append(generator.a[simulation.tick_counter])
            assert generator.b[simulation.tick_counter] > 0
            b.append(generator.b[simulation.tick_counter])
            fixed_cost.append(generator.fixed_cost)
            p_max.append(generator.p_max[simulation.tick_counter])
            p_min.append(generator.p_min[simulation.tick_counter])
            generator_locations.append(simulation.nodes.index(generator.node))

    """ Encode LSEs """
    demand = []
    nodes_lse = []
    for lse in simulation.lses:
        if not lse.is_detached:
            demand.append(lse.demand[simulation.tick_counter])
            nodes_lse.append(simulation.nodes.index(lse.node))

    """ Encode Transmission Lines """
    lines = []
    line_reactances = []
    line_capacities = []
    line_start_nodes = []
    line_end_nodes = []
    for line in simulation.lines:
        if not line.is_detached:
            lines.append(
                [
                    simulation.nodes.index(line.node1),
                    simulation.nodes.index(line.node2),
                ]
            )
            line_start_nodes.append(simulation.nodes.index(line.node1))
            line_end_nodes.append(simulation.nodes.index(line.node2))
            assert line.reactance >= 0
            line_reactances.append(line.reactance)
            assert line.max_capacity >= 0
            line_capacities.append(line.max_capacity)

    """ Create the dict """
    current_input = {
        # Generators
        "generator_a": np.array(a) * base_s,
        "generator_b": np.array(b) * base_s * base_s,
        "generator_locations": np.array(generator_locations),
        "generator_commitement_minimum": np.array(p_min) / base_s,
        "generator_commitement_maximum": np.array(p_max) / base_s,
        # Demands
        "demand": np.array(demand) / base_s,
        "demand_locations": np.array(nodes_lse),
        # Lines
        "line_start_nodes": np.array(line_start_nodes),
        "line_end_nodes": np.array(line_end_nodes),
        "line_reactances": np.array(line_reactances),
        "line_capacity": np.array(line_capacities) / base_s,
    }

    """ Pre-solve checks """
    # Check that the maximum capacity of the generators is higher than the minimum demand of the LSEs
    assert np.sum(p_max) >= np.sum(demand), f"{np.sum(p_max)} >= {np.sum(demand)}"
    # Check that the pmin of the the generators is lower than the maximum demand of the LSEs
    assert np.sum(p_min) <= np.sum(demand), f"Minimum Generation [{np.sum(p_min):,.2f}] <= Total Demand [{np.sum(demand):,.2f}]"
    """ Solve """
    results = DER.solver.solve(**current_input)
    """ Transform from PU """
    results["Commitements"] = np.round(results["Commitements"], 4) * base_s
    results["LMPs"] = np.round(results["LMPs"], 4) / base_s

    """ Store the results back into the simulation """
    # Store the generation to the generators
    for generator, generation_result in zip(
        [i for i in simulation.generators if not i.is_detached],
        results["Commitements"],
    ):
        generator.generation[simulation.tick_counter] = generation_result
    # lse_lmps
    for node, lmp in zip(simulation.nodes, results["LMPs"]):
        node.lmp[simulation.tick_counter] = lmp

    """ Flow in Transmission Lines """
    voltages = np.array([0, ] + list(results["Voltage Angles"]))
    voltages_degrees = voltages * 180 / np.pi
    for node, angle, angle_degrees in zip(simulation.nodes, voltages, voltages_degrees):
        node.voltage_angle[simulation.tick_counter] = angle
        node.voltage_angle_degrees[simulation.tick_counter] = angle_degrees
    
    for line in [i for i in simulation.lines if not i.is_detached]:
        line.flow[simulation.tick_counter] = round(
            (1 / line.reactance)
            * (line.node1.voltage_angle[simulation.tick_counter] - line.node2.voltage_angle[simulation.tick_counter])
            * base_s,
            4
        )
        assert line.flow[simulation.tick_counter] <= line.max_capacity + 0.01
