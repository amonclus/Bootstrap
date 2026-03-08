from __future__ import annotations
from typing import List, Dict
import networkx as nx
from simulation.bootstrap import BootstrapPercolation


def sweep_er_probability(n: int, probabilities: List[float], threshold: int = 2, num_trials: int = 50,) -> List[Dict]:

    results = []

    for p in probabilities:
        g = nx.erdos_renyi_graph(n, p)

        sim = BootstrapPercolation(g, threshold)
        metrics = sim.collect_metrics(num_trials=num_trials)

        results.append(
            {
                "graph_type": "erdos_renyi",
                "n": n,
                "p": p,
                "edges": g.number_of_edges(),
                "cascade_size": metrics.cascade_size,
                "cascade_probability": metrics.cascade_probability,
                "critical_seed_size": metrics.critical_seed_size,
                "time_to_cascade": metrics.time_to_cascade,
                "percolation_threshold": metrics.percolation_threshold,
            }
        )

    return results


def sweep_geometric_radius(n: int,radii: List[float], threshold: int = 2, num_trials: int = 50,) -> List[Dict]:

    results = []

    for r in radii:
        g = nx.random_geometric_graph(n, r)

        sim = BootstrapPercolation(g, threshold)
        metrics = sim.collect_metrics(num_trials=num_trials)

        results.append(
            {
                "graph_type": "random_geometric",
                "n": n,
                "radius": r,
                "edges": g.number_of_edges(),
                "cascade_size": metrics.cascade_size,
                "cascade_probability": metrics.cascade_probability,
                "critical_seed_size": metrics.critical_seed_size,
                "time_to_cascade": metrics.time_to_cascade,
                "percolation_threshold": metrics.percolation_threshold,
            }
        )

    return results


def sweep_lattice_size(sizes: List[int], threshold: int = 2, num_trials: int = 50,) -> List[Dict]:

    results = []

    for size in sizes:
        g = nx.grid_2d_graph(size, size)
        g = nx.convert_node_labels_to_integers(g)

        sim = BootstrapPercolation(g, threshold)
        metrics = sim.collect_metrics(num_trials=num_trials)

        results.append(
            {
                "graph_type": "lattice",
                "grid_size": size,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "cascade_size": metrics.cascade_size,
                "cascade_probability": metrics.cascade_probability,
                "critical_seed_size": metrics.critical_seed_size,
                "time_to_cascade": metrics.time_to_cascade,
                "percolation_threshold": metrics.percolation_threshold,
            }
        )

    return results


def sweep_seed_fraction(graph: nx.Graph, seed_fractions: List[float], threshold: int = 2, num_trials: int = 50,) -> List[Dict]:

    results = []

    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))

        sim = BootstrapPercolation(graph, threshold)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials
        )

        results.append(
            {
                "seed_fraction": frac,
                "seed_size": seed_size,
                "cascade_probability": prob,
                "cascade_size": avg_fraction,
                "time_to_cascade": avg_time,
            }
        )

    return results


def run_full_parameter_sweep(k_values: list[int] = [1, 2, 3, 4, 5], num_trials: int = 50) -> Dict[str, List[Dict]]:

    results = {}

    # Erdős–Rényi sweep
    er_probs = [0.01, 0.02, 0.05, 0.1, 0.2]
    results['erdos_renyi'] = []
    for k in k_values:
        for entry in sweep_er_probability(100, er_probs, threshold=k, num_trials=num_trials):
            entry['threshold'] = k
            results['erdos_renyi'].append(entry)

    # Random geometric sweep
    radii = [0.05, 0.1, 0.15, 0.2, 0.25]
    results['random_geometric'] = []
    for k in k_values:
        for entry in sweep_geometric_radius(100, radii, threshold=k, num_trials=num_trials):
            entry['threshold'] = k
            results['random_geometric'].append(entry)

    # Lattice sweep
    sizes = [5, 10, 15]
    results['lattice'] = []
    for k in k_values:
        for entry in sweep_lattice_size(sizes, threshold=k, num_trials=num_trials):
            entry['threshold'] = k
            results['lattice'].append(entry)

    return results