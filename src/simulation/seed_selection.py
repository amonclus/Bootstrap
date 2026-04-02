"""
Pluggable seed-selection strategies for contagion simulations.

All strategy functions share the same interface:
    select(graph: nx.Graph, n: int) -> list[int]

Use ``select_seeds`` to dispatch by strategy name.
"""

from __future__ import annotations

import random
from enum import Enum

import networkx as nx


class SeedStrategy(str, Enum):
    RANDOM = "random"
    HIGH_DEGREE = "high_degree"
    HIGH_KCORE = "high_kcore"


def random_seeds(graph: nx.Graph, n: int) -> list[int]:
    """Select *n* nodes uniformly at random."""
    return random.sample(list(graph.nodes()), n)


def high_degree_seeds(graph: nx.Graph, n: int) -> list[int]:
    """Select the *n* nodes with the highest degree."""
    sorted_nodes = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    return sorted_nodes[:n]



def high_kcore_seeds(graph: nx.Graph, n: int) -> list[int]:
    """Select the *n* nodes with the highest coreness (k-core number)."""
    coreness = nx.core_number(graph)
    sorted_nodes = sorted(graph.nodes(), key=lambda v: coreness[v], reverse=True)
    return sorted_nodes[:n]


def select_seeds(graph: nx.Graph, n: int, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> list[int]:
    """Dispatch to the requested seed-selection strategy.

    Args:
        graph: The network to draw seeds from.
        n: Number of seed nodes to select.
        strategy: One of the ``SeedStrategy`` values (or its string equivalent).

    Returns:
        A list of *n* node IDs chosen according to the strategy.
    """
    strategy = SeedStrategy(strategy)
    if strategy == SeedStrategy.RANDOM:
        return random_seeds(graph, n)
    if strategy == SeedStrategy.HIGH_DEGREE:
        return high_degree_seeds(graph, n)
    if strategy == SeedStrategy.HIGH_KCORE:
        return high_kcore_seeds(graph, n)
    raise ValueError(f"Unknown seed strategy: {strategy}")
