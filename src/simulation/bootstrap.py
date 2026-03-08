from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import random_seeds


@dataclass
class BootstrapResult:
    infected_nodes: set[int] = field(default_factory=set)
    cascade_size: int = 0
    cascade_fraction: float = 0.0
    time_to_cascade: int = 0
    is_full_cascade: bool = False


@dataclass
class PercolationMetrics:
    cascade_size: float = 0.0  # average cascade fraction (infected / total nodes)
    critical_seed_size: int = 0  # minimum seeds for full cascade
    cascade_probability: float = 0.0  # probability of full cascade
    time_to_cascade: float = 0.0  # average rounds to stabilize
    percolation_threshold: float = 0.0  # critical seed fraction


class BootstrapPercolation:

    def __init__(self, graph: nx.Graph, threshold: int = 2) -> None:
        self.graph = graph
        self.threshold = threshold
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[BootstrapResult, list[set]]:
        infected = set(seed_nodes)
        rounds = 0
        activation_sequence = []

        if record_sequence:
            activation_sequence.append(set(infected))  # initial seed round

        while True:
            newly_infected = set()
            for node in self.graph.nodes():
                if node in infected:
                    continue
                infected_neighbors = sum(
                    1 for neighbor in self.graph.neighbors(node) if neighbor in infected
                )
                if infected_neighbors >= self.threshold:
                    newly_infected.add(node)

            if not newly_infected:
                break

            infected |= newly_infected
            rounds += 1

            if record_sequence:
                activation_sequence.append(set(newly_infected))

        result = BootstrapResult(
            infected_nodes=infected,
            cascade_size=len(infected),
            cascade_fraction=len(infected) / self.n if self.n > 0 else 0.0,
            time_to_cascade=rounds,
            is_full_cascade=(len(infected) == self.n),
        )

        return result, activation_sequence if record_sequence else []

    def cascade_probability(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None,) \
            -> tuple[float, float, float]:

        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        full_cascades = 0
        total_fraction = 0.0
        total_time = 0

        for _ in range(num_trials):
            seed_nodes = random_seeds(self.graph, seed_size)
            result, _ = self.run(seed_nodes)
            total_fraction += result.cascade_fraction
            total_time += result.time_to_cascade
            if result.is_full_cascade:
                full_cascades += 1

        prob = full_cascades / num_trials
        avg_fraction = total_fraction / num_trials
        avg_time = total_time / num_trials
        return prob, avg_fraction, avg_time

    def find_critical_seed_size(self, num_trials: int = 50, cascade_threshold: float = 1.0, probability_threshold: float = 0.5, seed: Optional[int] = None, ) -> int:
        if seed is not None:
            random.seed(seed)

        low, high = 1, self.n
        result = self.n  # worst case: all nodes

        while low <= high:
            mid = (low + high) // 2
            prob, _, _ = self.cascade_probability(mid, num_trials)
            if prob >= probability_threshold:
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return result

    def find_percolation_threshold(self, num_trials: int = 50, probability_threshold: float = 0.5, seed: Optional[int] = None, ) -> float:
        critical = self.find_critical_seed_size(
            num_trials=num_trials,
            probability_threshold=probability_threshold,
            seed=seed,
        )
        return critical / self.n if self.n > 0 else 0.0

    def collect_metrics(self, seed_size: Optional[int] = None, num_trials: int = 100, seed: Optional[int] = None, ) -> PercolationMetrics:
        # Find critical seed size and percolation threshold
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed)

        percolation_thresh = critical_seed / self.n if self.n > 0 else 0.0

        # Use provided seed_size or fall back to critical
        eval_size = seed_size if seed_size is not None else critical_seed

        # Estimate cascade probability and averages at that seed size
        prob, avg_fraction, avg_time = self.cascade_probability(seed_size=eval_size, num_trials=num_trials, seed=seed)

        return PercolationMetrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            percolation_threshold=percolation_thresh,
        )
