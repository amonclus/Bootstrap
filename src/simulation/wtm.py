"""
WTM — Watts Threshold Model.

A node activates when the *fraction* of its neighbours that are infected reaches
a threshold φ (phi).  Unlike bootstrap percolation which uses an absolute count k,
WTM normalises by degree, making the threshold relative.

  P(activation) = 1  if  infected_neighbours / degree >= phi
                  0  otherwise

Nodes with degree 0 never activate (0/0 treated as 0).

The cascade is deterministic given the seed set.  Infected nodes do not recover
(same as bootstrap percolation).  Terminates when no new activations occur.

Limiting cases:
  - phi → 0  :  any infected neighbour activates you (like k=1 bootstrap)
  - phi → 1  :  all neighbours must be infected (very hard to cascade)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds


@dataclass
class WTMResult:
    """Stores the result of one WTM simulation run."""
    infected_nodes: set[int] = field(default_factory=set)
    cascade_size: int = 0
    cascade_fraction: float = 0.0
    time_to_cascade: int = 0
    is_full_cascade: bool = False


@dataclass
class WTMMetrics:
    """Stores the metrics collected across multiple WTM simulation trials."""
    cascade_size: float = 0.0          # average cascade fraction across trials
    critical_seed_size: int = 0        # minimum seeds for full cascade
    cascade_probability: float = 0.0   # probability of full cascade
    time_to_cascade: float = 0.0       # average rounds to stabilise
    percolation_threshold: float = 0.0 # critical seed fraction
    seed_strategy: str = SeedStrategy.RANDOM


class WTMModel:
    """
    Watts Threshold Model.

    Each round: a susceptible node activates if
        infected_neighbours / degree(node) >= phi.

    Terminates when no further activations are possible.
    """

    def __init__(self, graph: nx.Graph, phi: float = 0.3) -> None:
        self.graph = graph
        self.phi = phi
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[WTMResult, list[set]]:
        """
        Runs one WTM cascade starting from the given seed nodes.

        Returns:
            result: WTMResult
            activation_sequence: list of sets (one per round); empty if record_sequence is False.
        """
        infected = set(seed_nodes)
        rounds = 0
        activation_sequence = []

        if record_sequence:
            activation_sequence.append(set(infected))

        while True:
            newly_infected: set[int] = set()
            for node in self.graph.nodes():
                if node in infected:
                    continue
                deg = self.graph.degree(node)
                if deg == 0:
                    continue
                infected_neighbours = sum(
                    1 for nb in self.graph.neighbors(node) if nb in infected
                )
                if infected_neighbours / deg >= self.phi:
                    newly_infected.add(node)

            if not newly_infected:
                break

            infected |= newly_infected
            rounds += 1

            if record_sequence:
                activation_sequence.append(set(newly_infected))

        result = WTMResult(
            infected_nodes=infected,
            cascade_size=len(infected),
            cascade_fraction=len(infected) / self.n if self.n > 0 else 0.0,
            time_to_cascade=rounds,
            is_full_cascade=(len(infected) == self.n),
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
        Estimates cascade probability for a given seed size.

        Returns:
            prob: fraction of trials that produced a full cascade.
            avg_fraction: average cascade fraction across trials.
            avg_time: average rounds to stabilisation.
        """
        if seed is not None:
            random.seed(seed)

        full_cascades = 0
        total_fraction = 0.0
        total_time = 0

        for _ in range(num_trials):
            seed_nodes = select_seeds(self.graph, seed_size, strategy)
            result, _ = self.run(set(seed_nodes))
            total_fraction += result.cascade_fraction
            total_time += result.time_to_cascade
            if result.is_full_cascade:
                full_cascades += 1

        return full_cascades / num_trials, total_fraction / num_trials, total_time / num_trials

    def find_critical_seed_size(
        self,
        num_trials: int = 50,
        probability_threshold: float = 0.5,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> int:
        """Finds the minimum seed size for a full cascade using binary search."""
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
    ) -> WTMMetrics:
        """Collects all WTM metrics for the given seed size."""
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)
        percolation_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time = self.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return WTMMetrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            percolation_threshold=percolation_thresh,
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
        """Compute per-node influence on cascade spread under WTM dynamics."""
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
            full_cascades = 0
            total_time = 0

            for _ in range(num_trials):
                companions = set(random.sample(other_nodes, pick))
                seed_set = companions | {target}
                result, _ = self.run(seed_set)
                fractions.append(result.cascade_fraction)
                total_time += result.time_to_cascade
                if result.is_full_cascade:
                    full_cascades += 1

            avg_fraction = sum(fractions) / num_trials
            mean_sq = sum(f * f for f in fractions) / num_trials
            std_fraction = (mean_sq - avg_fraction ** 2) ** 0.5

            results.append({
                "node": target,
                "influence_score": round(avg_fraction, 6),
                "cascade_probability": round(full_cascades / num_trials, 4),
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
        """Assess how much each node's removal reduces cascade spread under WTM dynamics."""
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(1, int(seed_fraction * n))

        baseline_fractions: list[float] = []
        baseline_full = 0
        for _ in range(num_trials):
            s = set(random.sample(nodes, seed_size))
            res, _ = self.run(s)
            baseline_fractions.append(res.cascade_fraction)
            if res.is_full_cascade:
                baseline_full += 1
        baseline_avg = sum(baseline_fractions) / num_trials
        baseline_prob = baseline_full / num_trials

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

            sub_sim = WTMModel(sub, self.phi)
            sub_seed_size = max(1, min(seed_size, len(sub_nodes)))

            fracs: list[float] = []
            full = 0
            total_time = 0

            for _ in range(num_trials):
                s = set(random.sample(sub_nodes, sub_seed_size))
                res, _ = sub_sim.run(s)
                fracs.append(res.cascade_fraction)
                total_time += res.time_to_cascade
                if res.is_full_cascade:
                    full += 1

            avg_frac = sum(fracs) / num_trials
            prob_frac = full / num_trials

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
