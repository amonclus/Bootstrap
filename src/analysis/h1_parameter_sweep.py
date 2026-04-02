"""
Parameter sweep functions for the H1 OR-Hybrid contagion model.
"""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.H1 import H1Model
from simulation.seed_selection import SeedStrategy


def h1_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
    threshold: int = 2,
    beta: float = 0.3,
    gamma: float = 0.1,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over seed fractions on a fixed graph."""
    results = []
    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))
        sim = H1Model(graph, threshold=threshold, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "seed_fraction": frac,
            "seed_size": seed_size,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results


def h1_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
    threshold: int = 2,
    gamma: float = 0.1,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over β (transmission rate) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for beta in betas:
        sim = H1Model(graph, threshold=threshold, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "beta": beta,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results


def h1_sweep_threshold(
    graph: nx.Graph,
    thresholds: List[int],
    beta: float = 0.3,
    gamma: float = 0.1,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over bootstrap threshold k values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for k in thresholds:
        sim = H1Model(graph, threshold=k, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "threshold": k,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results
