"""
Parameter sweep functions for the H3 Probabilistic Threshold Hybrid model.
"""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.H3 import H3Model
from simulation.seed_selection import SeedStrategy


def h3_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
    beta: float = 0.3,
    gamma: float = 0.1,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over initial seed fractions on a fixed graph."""
    results = []
    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))
        sim = H3Model(graph, beta=beta, gamma=gamma)
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


def h3_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
    gamma: float = 0.1,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """
    Sweep over β values.  As β increases the soft infection threshold m* = 1/β
    decreases, transitioning the model from SIR-like to bootstrap-like behaviour.
    """
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for beta in betas:
        sim = H3Model(graph, beta=beta, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        soft_threshold = round(1.0 / beta, 2) if beta > 0 else float("inf")
        results.append({
            "beta": beta,
            "soft_threshold_m_star": soft_threshold,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results
