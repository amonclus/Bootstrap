"""
Parameter sweep functions for the SIR epidemic model.
Each function varies one parameter, runs multiple trials, and returns structured results.
Results are returned as lists of dicts suitable for saving as CSV/JSON before plotting.
"""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.sir import SIRModel


def sir_sweep_seed_fraction(
    graph: nx.Graph,
    fractions: List[float],
    beta: float,
    gamma: float,
    num_trials: int = 50,
) -> List[Dict]:
    """
    Sweeps over initial seed fractions on a fixed graph.

    Returns:
        List of dicts with keys: seed_fraction, seed_size, epidemic_size,
        epidemic_probability, time_to_epidemic.
    """
    results = []
    n = graph.number_of_nodes()

    for frac in fractions:
        seed_size = max(1, int(frac * n))
        sim = SIRModel(graph, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.epidemic_probability(seed_size, num_trials=num_trials)

        results.append({
            "seed_fraction": round(frac, 4),
            "seed_size": seed_size,
            "epidemic_size": round(avg_fraction, 4),
            "epidemic_probability": round(prob, 4),
            "time_to_epidemic": round(avg_time, 2),
        })

    return results


def sir_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
    gamma: float,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
) -> List[Dict]:
    """
    Sweeps over transmission rate β on a fixed graph.

    Returns:
        List of dicts with keys: beta, epidemic_size, epidemic_probability, time_to_epidemic.
    """
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for beta in betas:
        sim = SIRModel(graph, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.epidemic_probability(seed_size, num_trials=num_trials)

        results.append({
            "beta": round(beta, 4),
            "epidemic_size": round(avg_fraction, 4),
            "epidemic_probability": round(prob, 4),
            "time_to_epidemic": round(avg_time, 2),
        })

    return results


def sir_sweep_er_probability(
    n: int,
    probabilities: List[float],
    beta: float,
    gamma: float,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
) -> List[Dict]:
    """
    Sweeps over edge probability p for Erdős–Rényi graphs.

    Returns:
        List of dicts with keys: p, n, edges, epidemic_size, epidemic_probability,
        critical_seed_size, time_to_epidemic, epidemic_threshold.
    """
    results = []

    for p in probabilities:
        g = nx.erdos_renyi_graph(n, p)
        seed_size = max(1, int(seed_fraction * n))
        sim = SIRModel(g, beta=beta, gamma=gamma)
        metrics = sim.collect_metrics(seed_size=seed_size, num_trials=num_trials)

        results.append({
            "graph_type": "erdos_renyi",
            "n": n,
            "p": p,
            "edges": g.number_of_edges(),
            "epidemic_size": round(metrics.epidemic_size, 4),
            "epidemic_probability": round(metrics.epidemic_probability, 4),
            "critical_seed_size": metrics.critical_seed_size,
            "time_to_epidemic": round(metrics.time_to_epidemic, 2),
            "epidemic_threshold": round(metrics.epidemic_threshold, 4),
        })

    return results


def sir_sweep_lattice_size(
    sizes: List[int],
    beta: float,
    gamma: float,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
) -> List[Dict]:
    """
    Sweeps over lattice grid side lengths.

    Returns:
        List of dicts with keys: grid_size, n, edges, epidemic_size, epidemic_probability,
        critical_seed_size, time_to_epidemic, epidemic_threshold.
    """
    results = []

    for size in sizes:
        g = nx.grid_2d_graph(size, size)
        g = nx.convert_node_labels_to_integers(g)
        n = g.number_of_nodes()
        seed_size = max(1, int(seed_fraction * n))
        sim = SIRModel(g, beta=beta, gamma=gamma)
        metrics = sim.collect_metrics(seed_size=seed_size, num_trials=num_trials)

        results.append({
            "graph_type": "lattice",
            "grid_size": size,
            "n": n,
            "edges": g.number_of_edges(),
            "epidemic_size": round(metrics.epidemic_size, 4),
            "epidemic_probability": round(metrics.epidemic_probability, 4),
            "critical_seed_size": metrics.critical_seed_size,
            "time_to_epidemic": round(metrics.time_to_epidemic, 2),
            "epidemic_threshold": round(metrics.epidemic_threshold, 4),
        })

    return results
