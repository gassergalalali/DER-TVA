"""
plotting.py
Creates network plots using networkx. Useful for debugging. 
"""
import typing

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import DER

def plot_at_iteration(s, iteration_number, figsize=(10, 10), dpi=80):
    G = nx.Graph()
    for node in s.nodes:
        G.add_node(node.name)

    for line in s.lines:
        G.add_edge(
            line.node1.name, 
            line.node2.name,
            f_cap=abs(line.flow.max())/line.max_capacity,
            )
    plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    plt.tight_layout()
    plt.margins(0.3)
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(
        G, 
        pos=pos,
        ax=plt.gca(),
        node_size=50,
        node_color='k',
    )
    nx.draw_networkx_labels(
        G, 
        pos=pos,
        ax=plt.gca(),
        verticalalignment='top',
        labels={n.name: 
        f"\nNode {s.nodes.index(n)+1}: {n.name}\n"
        +f"({len(n.generators)} Generators + {len(n.lses)} LSEs)\n" 
        +f"(Generation = {sum([g.generation[iteration_number] for g in n.generators]):,.2f} MW)\n"
        +f"(Max Gen. Capacity = {sum([g.p_max[iteration_number] for g in n.generators]):,.2f} MW)\n"
        +f"(Min Gen. Capacity = {sum([g.p_min[iteration_number] for g in n.generators]):,.2f} MW)\n"
        +f"(Demand = {sum([l.demand[iteration_number] for l in n.lses]):,.2f} MW)\n" 
        +f"(LMP = {n.lmp[iteration_number]:,.2f} USD/MW)\n" 
        +f"(Residential Customers = {[l for l in n.lses if l.sector=='Residential'][0].number_of_active_customers[iteration_number]:,.0f})\n" 
        for n in s.nodes},
    )
    nx.draw_networkx_edges(
        G, 
        pos=pos,
        ax=plt.gca(),
        edge_color=[abs(line.flow[iteration_number])/line.max_capacity for line in s.lines],
        edge_cmap=plt.get_cmap("flare_r"),
        width=2,
        arrows=True,
    )
    nx.draw_networkx_edge_labels(
        G, 
        pos=pos,
        ax=plt.gca(),
        edge_labels={
            (line.node1.name, line.node2.name): 
            f"Line {s.lines.index(line)+1}\n"
            +f"Flow = {line.flow[iteration_number]:,.2f} MW\n"
            +f"Cap = {line.max_capacity:,.2f} MW\n"
            +f"Reactance = {line.reactance:,.2f} Ohm\n"
            +f"Flow/cap = ({(line.flow[iteration_number])/line.max_capacity:.2%})" 
            for line in s.lines},
    )
    print(G.edges)
    plt.axis('off')


def plot_network(
        simulation: DER.Simulation,
        sector: str = 'ANY',
        plot_detached=True,
        plot_generators=True,
        plot_lses=True
) -> None:
    """
    Plots the network...
    :param sector:
    :param simulation:
    :return: None
    """
    def _node_label(n):
        return f"Node {simulation.nodes.index(n) + 1}\n\n"

    def _lse_label(l):
        return f"LSE {simulation.lses.index(l) + 1}\n\n"

    G = nx.Graph()
    for node in simulation.nodes:
        node_label = _node_label(node)
        G.add_node(
            node_label,
            node=node,
            type="Node"
        )
        # add the generators
        if plot_generators:
            if plot_detached:
                list_of_generators = node.generators
            else:
                list_of_generators = [generator for generator in node.generators if not generator.is_detached]
            for generator in list_of_generators:
                generator_label = f"\n\nGenerator {simulation.generators.index(generator) + 1}"
                G.add_node(
                    generator_label,
                    generator=generator,
                    type="Generator"
                )
                G.add_edge(
                    node_label,
                    generator_label,
                    weights=0.4
                )
        # add the LSESs
        if plot_lses:
            if sector == "ANY":
                list_of_lses = node.lses
            else:
                if sector not in DER.DemandFunctions.sectors:
                    raise ValueError(f"Unknown sector '{sector}'")
                else:
                    if plot_detached:
                        list_of_lses = [lse for lse in node.lses if lse.sector == sector]
                    else:
                        list_of_lses = [lse for lse in node.lses if lse.sector == sector and not lse.is_detached]
            for lse in list_of_lses:
                lse_label = _lse_label(lse)
                G.add_node(
                    lse_label,
                    type='LSE'
                )
                G.add_edge(
                    node_label,
                    lse_label,
                    lse=lse,
                    weights=0.4
                )

    # Add the Tranmission Lines
    for i, line in enumerate(simulation.lines):
        for node1 in simulation.nodes:
            if line.node1 == node1:
                for node2 in simulation.nodes:
                    if line.node2 == node2:
                        G.add_edge(
                            _node_label(node1),
                            _node_label(node2),
                            node1=node2,
                            node2=node2,
                            type='Main',
                            line=line,
                            label=f"Line {i + 1}",
                            weights=1
                        )
    pos = nx.kamada_kawai_layout(
        G,
        scale=0.9
    )

    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()
    nx.draw_networkx(
        G=G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=[
            'blue' if node['type'] == 'Node'
            else 'green' if node['type'] == 'LSE'
            else 'red' if node['type'] == 'Generator'
            else 'grey'
            for node in G.nodes.values()
        ],
        node_size=100,
        font_size=10,
        width=[
            3 if 'node1' in G.edges[edge].keys() else 1
            for edge in G.edges
        ],
        style=[
            'solid' if 'node1' in G.edges[edge].keys() else 'dashed'
            for edge in G.edges
        ]
    )
    nx.draw_networkx_edge_labels(
        G=G,
        ax=ax,
        pos=pos,
        edge_labels={
            (n1, n2): G.edges[n1, n2]['label'] for n1, n2
            in G.edges
            if 'label' in G.edges[n1, n2].keys()
        }
    )
    return fig


