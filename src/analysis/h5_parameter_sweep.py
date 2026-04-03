"""Parameter sweep functions for the H5 Sequential Hybrid (SIS → WTM) model."""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.H5 import H5Model
from simulation.seed_selection import SeedStrategy


def h5_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
    phi: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.1,
    switch_fraction: float = 0.2,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over seed fractions on a fixed graph."""
    results = []
    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))
        sim = H5Model(graph, phi=phi, beta=beta, gamma=gamma, switch_fraction=switch_fraction)
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
        })

    return results


def h5_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
    phi: float = 0.3,
    gamma: float = 0.1,
    switch_fraction: float = 0.2,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over β (transmission rate) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for beta in betas:
        sim = H5Model(graph, phi=phi, beta=beta, gamma=gamma, switch_fraction=switch_fraction)
        prob, avg_fraction, avg_time, switch_prob, _ = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "beta": beta,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
            "switch_probability": switch_prob,
        })

    return results


def h5_sweep_phi(
    graph: nx.Graph,
    phis: List[float],
    beta: float = 0.3,
    gamma: float = 0.1,
    switch_fraction: float = 0.2,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over φ (WTM fractional threshold) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for phi in phis:
        sim = H5Model(graph, phi=phi, beta=beta, gamma=gamma, switch_fraction=switch_fraction)
        prob, avg_fraction, avg_time, switch_prob, _ = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "phi": phi,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
            "switch_probability": switch_prob,
        })

    return results
