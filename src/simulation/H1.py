"""
H1 — OR-Hybrid contagion model.

A susceptible node becomes infected each round if either:
  (1) the SIR channel fires: at least one infected neighbour transmits with probability β, OR
  (2) the bootstrap channel fires: the node has ≥ threshold infected neighbours simultaneously.

Infected nodes recover at rate γ per round (same as SIR).
Terminates when no infected nodes remain.

Limiting cases:
  - threshold → ∞  :  reduces to pure SIR
  - β → 0, γ → 0  :  reduces to pure bootstrap percolation
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds

_LARGE_CASCADE_FRACTION = 0.5


@dataclass
class H1Result:
    """Stores the result of one H1 simulation run."""
    infected_nodes: set[int] = field(default_factory=set)   # all nodes ever infected (I ∪ R)
    recovered_nodes: set[int] = field(default_factory=set)  # R nodes at end
    cascade_size: int = 0
    cascade_fraction: float = 0.0
    time_to_cascade: int = 0
    is_large_cascade: bool = False


@dataclass
class H1Metrics:
    """Stores the metrics collected across multiple H1 simulation trials."""
    cascade_size: float = 0.0          # average cascade fraction across trials
    critical_seed_size: int = 0        # minimum seeds for a large cascade
    cascade_probability: float = 0.0   # fraction of trials that produced a large cascade
    time_to_cascade: float = 0.0       # average rounds to stabilise
    cascade_threshold: float = 0.0     # critical_seed_size / n
    seed_strategy: str = SeedStrategy.RANDOM


class H1Model:
    """
    OR-Hybrid contagion model (H1).

    Each round:
      1. Infection step: a susceptible node becomes infected if EITHER
         - the SIR channel fires (any infected neighbour transmits with prob β), OR
         - it has ≥ threshold infected neighbours (bootstrap channel).
      2. Recovery step: each currently infected node (those infected before this round)
         recovers independently with probability γ.

    Terminates when no infected nodes remain.
    """

    def __init__(self, graph: nx.Graph, threshold: int, beta: float, gamma: float) -> None:
        self.graph = graph
        self.threshold = threshold
        self.beta = beta
        self.gamma = gamma
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[H1Result, list]:
        """
        Runs one H1 simulation starting from the given seed nodes.

        Args:
            seed_nodes: Initial set of infected nodes.
            record_sequence: If True, records (newly_infected, newly_recovered) per round
                             in the same format as the SIR model, compatible with animate_cascade.

        Returns:
            result: H1Result
            activation_sequence: list of (set, set) tuples; empty if record_sequence is False.
        """
        infected = set(seed_nodes)
        susceptible = set(self.graph.nodes()) - infected
        recovered: set[int] = set()
        ever_infected = set(seed_nodes)
        rounds = 0
        activation_sequence: list = []

        if record_sequence:
            activation_sequence.append((set(infected), set()))

        while infected:
            newly_infected: set[int] = set()

            # SIR channel: each infected node tries to infect each susceptible neighbour
            for node in infected:
                for neighbour in self.graph.neighbors(node):
                    if neighbour in susceptible and random.random() < self.beta:
                        newly_infected.add(neighbour)

            # Bootstrap channel: susceptible nodes with ≥ threshold infected neighbours
            for node in susceptible:
                if node not in newly_infected:
                    infected_neighbour_count = sum(
                        1 for nb in self.graph.neighbors(node) if nb in infected
                    )
                    if infected_neighbour_count >= self.threshold:
                        newly_infected.add(node)

            # Recovery step (applies to nodes infected before this round)
            newly_recovered = {node for node in infected if random.random() < self.gamma}

            susceptible -= newly_infected
            infected -= newly_recovered
            infected |= newly_infected
            recovered |= newly_recovered
            ever_infected |= newly_infected
            rounds += 1

            if record_sequence and (newly_infected or newly_recovered):
                activation_sequence.append((set(newly_infected), set(newly_recovered)))

        cascade_size = len(ever_infected)
        cascade_fraction = cascade_size / self.n if self.n > 0 else 0.0

        result = H1Result(
            infected_nodes=ever_infected,
            recovered_nodes=recovered,
            cascade_size=cascade_size,
            cascade_fraction=cascade_fraction,
            time_to_cascade=rounds,
            is_large_cascade=(cascade_fraction >= _LARGE_CASCADE_FRACTION),
        )

        return result, activation_sequence if record_sequence else []

    def cascade_probability(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> tuple[float, float, float]:
        """
        Estimates the probability of a large cascade for a given seed size.

        Returns:
            prob: fraction of trials that produced a large cascade.
            avg_fraction: average cascade fraction across all trials.
            avg_time: average rounds to stabilisation.
        """
        if seed is not None:
            random.seed(seed)

        large_cascades = 0
        total_fraction = 0.0
        total_time = 0

        for _ in range(num_trials):
            seed_nodes = set(select_seeds(self.graph, seed_size, strategy))
            result, _ = self.run(seed_nodes)
            total_fraction += result.cascade_fraction
            total_time += result.time_to_cascade
            if result.is_large_cascade:
                large_cascades += 1

        return large_cascades / num_trials, total_fraction / num_trials, total_time / num_trials

    def find_critical_seed_size(self, num_trials: int = 50, probability_threshold: float = 0.5, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> int:
        """
        Finds the minimum seed size for a large cascade with at least probability_threshold
        probability, using binary search.
        """
        if seed is not None:
            random.seed(seed)

        low, high = 1, self.n
        result = self.n

        while low <= high:
            mid = (low + high) // 2
            prob, _, _ = self.cascade_probability(mid, num_trials, strategy=strategy)
            if prob >= probability_threshold:
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return result

    def collect_metrics(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> H1Metrics:
        """Collects all H1 metrics for the given seed size."""
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)
        cascade_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time = self.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return H1Metrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            cascade_threshold=cascade_thresh,
            seed_strategy=SeedStrategy(strategy).value,
        )

    # ── Node vulnerability analysis ─────────────────────────────────────

    def node_influence_analysis(self, seed_fraction: float = 0.05, num_trials: int = 20, seed: Optional[int] = None, progress_callback=None) -> list[dict]:
        """Compute per-node influence on cascade spread.

        For every node v the method runs num_trials H1 simulations where v is always
        part of the initial seed set (remaining seeds chosen uniformly at random).
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(2, int(seed_fraction * n))

        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)

        results: list[dict] = []

        for idx, target in enumerate(nodes):
            other_nodes = [nd for nd in nodes if nd != target]
            pick = min(seed_size - 1, len(other_nodes))

            fractions: list[float] = []
            large_cascades = 0
            total_time = 0

            for _ in range(num_trials):
                companions = set(random.sample(other_nodes, pick))
                seed_set = companions | {target}
                result, _ = self.run(seed_set)
                fractions.append(result.cascade_fraction)
                total_time += result.time_to_cascade
                if result.is_large_cascade:
                    large_cascades += 1

            avg_fraction = sum(fractions) / num_trials
            mean_sq = sum(f * f for f in fractions) / num_trials
            std_fraction = (mean_sq - avg_fraction ** 2) ** 0.5

            results.append({
                "node": target,
                "influence_score": round(avg_fraction, 6),
                "cascade_probability": round(large_cascades / num_trials, 4),
                "avg_time": round(total_time / num_trials, 2),
                "cascade_std": round(std_fraction, 6),
                "degree": self.graph.degree(target),
                "betweenness": round(betweenness[target], 6),
                "closeness": round(closeness[target], 6),
            })

            if progress_callback is not None:
                progress_callback(idx + 1, n)

        results.sort(key=lambda r: r["influence_score"], reverse=True)
        return results

    # ── Node blocking / immunisation analysis ───────────────────────────

    def node_blocking_analysis(self, seed_fraction: float = 0.05, num_trials: int = 20, seed: Optional[int] = None, progress_callback=None) -> tuple[list[dict], float, float]:
        """Assess how much each node's removal reduces cascade spread.

        For every node v the method builds a subgraph G − {v}, runs num_trials H1
        simulations on it, and compares the result to a baseline on the original graph.
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(1, int(seed_fraction * n))

        # Baseline on the full graph
        baseline_fractions: list[float] = []
        baseline_large = 0
        for _ in range(num_trials):
            s = set(random.sample(nodes, seed_size))
            res, _ = self.run(s)
            baseline_fractions.append(res.cascade_fraction)
            if res.is_large_cascade:
                baseline_large += 1
        baseline_avg = sum(baseline_fractions) / num_trials
        baseline_prob = baseline_large / num_trials

        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)

        results: list[dict] = []

        for idx, target in enumerate(nodes):
            sub = self.graph.copy()
            sub.remove_node(target)
            sub_nodes = list(sub.nodes())

            if len(sub_nodes) == 0:
                results.append({
                    "node": target,
                    "cascade_reduction": round(baseline_avg, 6),
                    "prob_reduction": round(baseline_prob, 4),
                    "cascade_blocked": 0.0,
                    "prob_blocked": 0.0,
                    "time_blocked": 0.0,
                    "degree": self.graph.degree(target),
                    "betweenness": round(betweenness[target], 6),
                    "closeness": round(closeness[target], 6),
                })
                if progress_callback is not None:
                    progress_callback(idx + 1, n)
                continue

            sub_sim = H1Model(sub, self.threshold, self.beta, self.gamma)
            sub_seed_size = max(1, min(seed_size, len(sub_nodes)))

            fracs: list[float] = []
            large = 0
            total_time = 0

            for _ in range(num_trials):
                s = set(random.sample(sub_nodes, sub_seed_size))
                res, _ = sub_sim.run(s)
                fracs.append(res.cascade_fraction)
                total_time += res.time_to_cascade
                if res.is_large_cascade:
                    large += 1

            avg_frac = sum(fracs) / num_trials
            prob_frac = large / num_trials

            results.append({
                "node": target,
                "cascade_reduction": round(baseline_avg - avg_frac, 6),
                "prob_reduction": round(baseline_prob - prob_frac, 4),
                "cascade_blocked": round(avg_frac, 6),
                "prob_blocked": round(prob_frac, 4),
                "time_blocked": round(total_time / num_trials, 2),
                "degree": self.graph.degree(target),
                "betweenness": round(betweenness[target], 6),
                "closeness": round(closeness[target], 6),
            })

            if progress_callback is not None:
                progress_callback(idx + 1, n)

        results.sort(key=lambda r: r["cascade_reduction"], reverse=True)
        return results, baseline_avg, baseline_prob
