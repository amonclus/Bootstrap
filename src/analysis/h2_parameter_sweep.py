"""
Parameter sweep functions for the H2 Sequential Hybrid (Switching Model).
"""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.H2 import H2Model
from simulation.seed_selection import SeedStrategy


def h2_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
    threshold: int = 2,
    beta: float = 0.3,
    gamma: float = 0.1,
    switch_fraction: float = 0.2,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over initial seed fractions on a fixed graph."""
    results = []
    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))
        sim = H2Model(graph, threshold=threshold, beta=beta, gamma=gamma, switch_fraction=switch_fraction)
        prob, avg_fraction, avg_time, switch_prob, avg_switch_frac = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "seed_fraction": frac,
            "seed_size": seed_size,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
            "switch_probability": switch_prob,
            "avg_switch_fraction": avg_switch_frac,
        })

    return results


def h2_sweep_switch_fraction(
    graph: nx.Graph,
    switch_fractions: List[float],
    threshold: int = 2,
    beta: float = 0.3,
    gamma: float = 0.1,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over the switch threshold f — the key parameter of H2."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for f in switch_fractions:
        sim = H2Model(graph, threshold=threshold, beta=beta, gamma=gamma, switch_fraction=f)
        prob, avg_fraction, avg_time, switch_prob, avg_switch_frac = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "switch_fraction": f,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
            "switch_probability": switch_prob,
            "avg_switch_fraction": avg_switch_frac,
        })

    return results


def h2_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
    threshold: int = 2,
    gamma: float = 0.1,
    switch_fraction: float = 0.2,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over β (SIR transmission rate) on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for beta in betas:
        sim = H2Model(graph, threshold=threshold, beta=beta, gamma=gamma, switch_fraction=switch_fraction)
        prob, avg_fraction, avg_time, switch_prob, avg_switch_frac = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "beta": beta,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
            "switch_probability": switch_prob,
            "avg_switch_fraction": avg_switch_frac,
        })

    return results
