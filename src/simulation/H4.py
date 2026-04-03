"""
H4 — OR-Hybrid: SIS + Watts Threshold Model.

A susceptible node becomes infected each round if either:
  (1) the SIS channel fires: at least one infected neighbour transmits with probability β, OR
  (2) the WTM channel fires: the fraction of infected neighbours >= φ.

Infected nodes recover at rate γ and return to susceptible (SIS dynamics — no permanent
immunity).  Terminates when no infected nodes remain or max_steps is reached.

This extends H1 (SIS/Bootstrap OR-hybrid) by replacing the absolute-count bootstrap
threshold with a fractional WTM threshold φ, making the reinforcement channel
degree-normalised.

Limiting cases:
  - phi → 1, γ → 0 : approaches pure SIS (WTM channel rarely fires)
  - β → 0, γ → 0   : approaches pure WTM
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds

_LARGE_CASCADE_FRACTION = 0.5
_DEFAULT_MAX_STEPS = 500


@dataclass
class H4Result:
    """Stores the result of one H4 simulation run."""
    infected_nodes: set[int] = field(default_factory=set)   # all nodes ever infected
    peak_infected: int = 0
    cascade_fraction: float = 0.0   # peak infected fraction
    cascade_size: int = 0           # = peak_infected
    time_to_cascade: int = 0
    is_large_cascade: bool = False
    infected_series: list = field(default_factory=list)


@dataclass
class H4Metrics:
    """Stores the metrics collected across multiple H4 simulation trials."""
    cascade_size: float = 0.0
    critical_seed_size: int = 0
    cascade_probability: float = 0.0
    time_to_cascade: float = 0.0
    cascade_threshold: float = 0.0
    seed_strategy: str = SeedStrategy.RANDOM


class H4Model:
    """
    OR-Hybrid SIS + WTM contagion model (H4).

    Each round:
      1. Infection step: a susceptible node is infected if EITHER
         - the SIS channel fires (any infected neighbour transmits with prob β), OR
         - it has infected_neighbours / degree >= phi (WTM channel).
      2. Recovery step: infected nodes recover with probability γ → susceptible.

    Terminates when no infected nodes remain or max_steps is reached.
    """

    def __init__(
        self,
        graph: nx.Graph,
        phi: float,
        beta: float,
        gamma: float,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self.graph = graph
        self.phi = phi
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_steps
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[H4Result, list]:
        """
        Runs one H4 simulation starting from the given seed nodes.

        Returns:
            result: H4Result
            activation_sequence: list of (set, set) per round; empty if record_sequence is False.
        """
        infected = set(seed_nodes)
        susceptible = set(self.graph.nodes()) - infected
        ever_infected = set(seed_nodes)
        rounds = 0
        infected_series = [len(infected)]
        peak_infected = len(infected)
        activation_sequence: list = []

        if record_sequence:
            activation_sequence.append((set(infected), set()))

        while infected and rounds < self.max_steps:
            newly_infected: set[int] = set()

            # SIS channel
            for node in infected:
                for neighbour in self.graph.neighbors(node):
                    if neighbour in susceptible and random.random() < self.beta:
                        newly_infected.add(neighbour)

            # WTM channel
            for node in susceptible:
                if node in newly_infected:
                    continue
                deg = self.graph.degree(node)
                if deg == 0:
                    continue
                infected_count = sum(1 for nb in self.graph.neighbors(node) if nb in infected)
                if infected_count / deg >= self.phi:
                    newly_infected.add(node)

            newly_recovered = {node for node in infected if random.random() < self.gamma}

            susceptible -= newly_infected
            susceptible |= newly_recovered
            infected -= newly_recovered
            infected |= newly_infected
            ever_infected |= newly_infected
            rounds += 1

            infected_series.append(len(infected))
            if len(infected) > peak_infected:
                peak_infected = len(infected)

            if record_sequence and (newly_infected or newly_recovered):
                activation_sequence.append((set(newly_infected), set(newly_recovered)))

        peak_fraction = peak_infected / self.n if self.n > 0 else 0.0

        result = H4Result(
            infected_nodes=ever_infected,
            peak_infected=peak_infected,
            cascade_fraction=peak_fraction,
            cascade_size=peak_infected,
            time_to_cascade=rounds,
            is_large_cascade=(peak_fraction >= _LARGE_CASCADE_FRACTION),
            infected_series=infected_series,
        )

        return result, activation_sequence if record_sequence else []

    def cascade_probability(
        self,
        seed_size: int,
        num_trials: int = 100,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> tuple[float, float, float]:
        """
        Estimates cascade statistics for a given seed size.

        Returns:
            prob: fraction of trials that produced a large cascade.
            avg_fraction: average peak cascade fraction.
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

    def find_critical_seed_size(
        self,
        num_trials: int = 50,
        probability_threshold: float = 0.5,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> int:
        """Finds the minimum seed size for a large cascade using binary search."""
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

    def collect_metrics(
        self,
        seed_size: int,
        num_trials: int = 100,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> H4Metrics:
        """Collects all H4 metrics for the given seed size."""
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)
        cascade_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time = self.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return H4Metrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            cascade_threshold=cascade_thresh,
            seed_strategy=SeedStrategy(strategy).value,
        )

    # ── Node vulnerability analysis ─────────────────────────────────────

    def node_influence_analysis(
        self,
        seed_fraction: float = 0.05,
        num_trials: int = 20,
        seed: Optional[int] = None,
        progress_callback=None,
    ) -> list[dict]:
        """Compute per-node influence on cascade spread under H4 dynamics."""
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

    def node_blocking_analysis(
        self,
        seed_fraction: float = 0.05,
        num_trials: int = 20,
        seed: Optional[int] = None,
        progress_callback=None,
    ) -> tuple[list[dict], float, float]:
        """Assess how much each node's removal reduces cascade spread under H4 dynamics."""
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(1, int(seed_fraction * n))

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

            sub_sim = H4Model(sub, self.phi, self.beta, self.gamma, self.max_steps)
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
