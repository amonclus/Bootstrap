"""Parameter sweep functions for the SIS epidemic model."""
from __future__ import annotations

from typing import List, Dict

import networkx as nx

from simulation.sis import SISModel
from simulation.seed_selection import SeedStrategy


def sis_sweep_seed_fraction(
    graph: nx.Graph,
    seed_fractions: List[float],
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
        sim = SISModel(graph, beta=beta, gamma=gamma)
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


def sis_sweep_beta(
    graph: nx.Graph,
    betas: List[float],
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
        sim = SISModel(graph, beta=beta, gamma=gamma)
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


def sis_sweep_gamma(
    graph: nx.Graph,
    gammas: List[float],
    beta: float = 0.3,
    seed_fraction: float = 0.05,
    num_trials: int = 50,
    strategy: SeedStrategy | str = SeedStrategy.RANDOM,
) -> List[Dict]:
    """Sweep over γ (recovery rate) values on a fixed graph."""
    results = []
    n = graph.number_of_nodes()
    seed_size = max(1, int(seed_fraction * n))

    for gamma in gammas:
        sim = SISModel(graph, beta=beta, gamma=gamma)
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
