"""Parameter sweep functions for the H6 Probabilistic Threshold (Soft WTM) model."""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.H6 import H6Model
from simulation.seed_selection import SeedStrategy


def h6_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
    phi: float = 0.3,
    gamma: float = 0.1,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over seed fractions on a fixed graph."""
    results = []
    n = graph.number_of_nodes()

    for frac in seed_fractions:
        seed_size = max(1, int(frac * n))
        sim = H6Model(graph, phi=phi, gamma=gamma)
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


def h6_sweep_phi(
    graph: nx.Graph,
    phis: List[float],
    gamma: float = 0.1,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over φ (fractional threshold) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for phi in phis:
        sim = H6Model(graph, phi=phi, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "phi": phi,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results


def h6_sweep_gamma(
    graph: nx.Graph,
    gammas: List[float],
    phi: float = 0.3,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over γ (recovery rate) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for gamma in gammas:
        sim = H6Model(graph, phi=phi, gamma=gamma)
        prob, avg_fraction, avg_time = sim.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, strategy=strategy
        )
        results.append({
            "gamma": gamma,
            "cascade_probability": prob,
            "cascade_size": avg_fraction,
            "time_to_cascade": avg_time,
        })

    return results
