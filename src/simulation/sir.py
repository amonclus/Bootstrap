"""
Algorithm module responsible for implementing the SIR (Susceptible-Infected-Recovered) epidemic
model and collecting all experimental metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds

_LARGE_EPIDEMIC_FRACTION = 0.5  # fraction of nodes that must be infected to count as a large epidemic


@dataclass
class SIRResult:
    """
    Stores the result of one SIR simulation run.
    """
    infected_nodes: set[int] = field(default_factory=set)   # all nodes ever infected (I ∪ R)
    recovered_nodes: set[int] = field(default_factory=set)  # R nodes at end of simulation
    epidemic_size: int = 0                                   # total nodes ever infected
    epidemic_fraction: float = 0.0                          # epidemic_size / n
    time_to_epidemic: int = 0                               # rounds until infected set is empty
    is_large_epidemic: bool = False                         # epidemic_fraction >= _LARGE_EPIDEMIC_FRACTION


@dataclass
class SIREpidemicMetrics:
    """
    Stores the metrics collected across multiple SIR simulation trials.
    """
    epidemic_size: float = 0.0          # average epidemic fraction across trials
    critical_seed_size: int = 0         # minimum seeds for a large epidemic
    epidemic_probability: float = 0.0   # fraction of trials that produced a large epidemic
    time_to_epidemic: float = 0.0       # average rounds to stabilise
    epidemic_threshold: float = 0.0     # critical_seed_size / n
    seed_strategy: str = SeedStrategy.RANDOM  # strategy used to select initial seeds


class SIRModel:
    """
    Class responsible for implementing the discrete-time SIR epidemic model.

    Each round:
      1. Infection step: every susceptible neighbour of an infected node becomes infected
         independently with probability beta.
      2. Recovery step: every currently infected node (those infected before this round)
         recovers independently with probability gamma.

    Terminates when no infected nodes remain.

    Special cases
    -------------
    - beta → 0 : no spreading; only the seed nodes ever get infected.
    - gamma → 1 : all infected nodes recover every round; epidemic dies quickly.
    """

    def __init__(self, graph: nx.Graph, beta: float, gamma: float) -> None:
        self.graph = graph
        self.beta = beta
        self.gamma = gamma
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[SIRResult, list[set]]:
        """
        Runs one SIR simulation starting from the given seed nodes.

        Args:
            seed_nodes: set[int]
                Initial set of infected nodes.
            record_sequence: bool
                If True, records the set of newly infected nodes per round for later display.

        Returns:
            result : SIRResult
            activation_sequence : list[set]
                Empty list when record_sequence is False.
        """
        infected = set(seed_nodes)
        susceptible = set(self.graph.nodes()) - infected
        recovered: set[int] = set()                                 #Also includes deaths
        ever_infected = set(seed_nodes)
        rounds = 0
        activation_sequence: list[set] = []

        if record_sequence:
            activation_sequence.append((set(infected), set()))  # initial seed round

        while infected:
            # --- infection step -------------------------------------------------
            newly_infected: set[int] = set()
            for node in infected:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in susceptible and random.random() < self.beta:
                        newly_infected.add(neighbor)

            # --- recovery step (applies to nodes infected before this round) ----
            newly_recovered = {node for node in infected if random.random() < self.gamma}

            susceptible -= newly_infected
            infected -= newly_recovered
            infected |= newly_infected
            recovered |= newly_recovered
            ever_infected |= newly_infected
            rounds += 1

            if record_sequence and (newly_infected or newly_recovered):
                activation_sequence.append((set(newly_infected), set(newly_recovered)))

        epidemic_size = len(ever_infected)
        epidemic_fraction = epidemic_size / self.n if self.n > 0 else 0.0

        result = SIRResult(
            infected_nodes=ever_infected,
            recovered_nodes=recovered,
            epidemic_size=epidemic_size,
            epidemic_fraction=epidemic_fraction,
            time_to_epidemic=rounds,
            is_large_epidemic=(epidemic_fraction >= _LARGE_EPIDEMIC_FRACTION),
        )

        return result, activation_sequence if record_sequence else []

    def epidemic_probability(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> tuple[float, float, float]:
        """
        Estimates the probability of a large epidemic for a given seed size.

        Args:
            seed_size: int
                Number of initially infected nodes.
            num_trials: int
                Number of independent trials.
            seed: int
                Optional random seed for reproducibility.

        Returns:
            prob : float
                Fraction of trials that produced a large epidemic.
            avg_fraction : float
                Average epidemic fraction across all trials.
            avg_time : float
                Average rounds to stabilisation across all trials.
        """
        if seed is not None:
            random.seed(seed)

        large_epidemics = 0
        total_fraction = 0.0
        total_time = 0

        for _ in range(num_trials):
            seed_nodes = set(select_seeds(self.graph, seed_size, strategy))
            result, _ = self.run(seed_nodes)
            total_fraction += result.epidemic_fraction
            total_time += result.time_to_epidemic
            if result.is_large_epidemic:
                large_epidemics += 1

        prob = large_epidemics / num_trials
        avg_fraction = total_fraction / num_trials
        avg_time = total_time / num_trials
        return prob, avg_fraction, avg_time

    def find_critical_seed_size(self, num_trials: int = 50, epidemic_threshold: float = _LARGE_EPIDEMIC_FRACTION, probability_threshold: float = 0.5, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> int:
        """
        Finds the minimum seed size required to trigger a large epidemic with at least
        probability_threshold probability, using binary search.

        Args:
            num_trials: int
                Trials per seed size evaluated during binary search.
            epidemic_threshold: float
                Fraction of nodes that must be infected to count as a large epidemic.
            probability_threshold: float
                Minimum probability of a large epidemic required to consider the seed size critical.
            seed: int
                Optional random seed for reproducibility.

        Returns:
            result : int
                The critical seed size.
        """
        if seed is not None:
            random.seed(seed)

        low, high = 1, self.n
        result = self.n  # worst case: all nodes

        while low <= high:
            mid = (low + high) // 2
            prob, _, _ = self.epidemic_probability(mid, num_trials, strategy=strategy)
            if prob >= probability_threshold:
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return result

    def find_epidemic_threshold(self, num_trials: int = 50, probability_threshold: float = 0.5, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> float:
        """
        Finds the epidemic threshold as the critical seed fraction (critical seed size / n).

        Args:
            num_trials: int
                Trials per seed size during binary search.
            probability_threshold: float
                Minimum probability of a large epidemic required.
            seed: int
                Optional random seed for reproducibility.

        Returns:
            result : float
                Critical seed size divided by total number of nodes.
        """
        critical = self.find_critical_seed_size(
            num_trials=num_trials,
            probability_threshold=probability_threshold,
            seed=seed,
            strategy=strategy,
        )
        return critical / self.n if self.n > 0 else 0.0

    def collect_metrics(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> SIREpidemicMetrics:
        """
        Collects all epidemic metrics for the given seed size.

        Args:
            seed_size: int
                Number of initially infected nodes used to evaluate epidemic probability
                and average epidemic size.
            num_trials: int
                Number of independent trials.
            seed: int
                Optional random seed for reproducibility.

        Returns:
            result : SIREpidemicMetrics
        """
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)
        epidemic_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time = self.epidemic_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return SIREpidemicMetrics(
            epidemic_size=avg_fraction,
            critical_seed_size=critical_seed,
            epidemic_probability=prob,
            time_to_epidemic=avg_time,
            epidemic_threshold=epidemic_thresh,
            seed_strategy=SeedStrategy(strategy).value,
        )

    # ── Node vulnerability analysis ─────────────────────────────────────

    def node_influence_analysis(self, seed_fraction: float = 0.05, num_trials: int = 20, seed: Optional[int] = None, progress_callback=None) -> list[dict]:
        """Compute per-node influence on epidemic spread.

        For every node *v* in the graph the method runs *num_trials* SIR
        simulations where *v* is always part of the initial seed set (the
        remaining seeds are chosen uniformly at random).

        Metrics collected per node
        * **influence_score** – average epidemic fraction across trials.
        * **cascade_probability** – fraction of trials that produced a large epidemic.
        * **avg_time** – average number of rounds until stabilisation.
        * **cascade_std** – standard deviation of the epidemic fraction.

        A high influence score and high cascade_probability means the node
        is a weak point.  A low influence score means it is resilient.
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
            large_epidemics = 0
            total_time = 0

            for _ in range(num_trials):
                companions = set(random.sample(other_nodes, pick))
                seed_set = companions | {target}
                result, _ = self.run(seed_set)
                fractions.append(result.epidemic_fraction)
                total_time += result.time_to_epidemic
                if result.is_large_epidemic:
                    large_epidemics += 1

            avg_fraction = sum(fractions) / num_trials
            mean_sq = sum(f * f for f in fractions) / num_trials
            std_fraction = (mean_sq - avg_fraction ** 2) ** 0.5

            results.append({
                "node": target,
                "influence_score": round(avg_fraction, 6),
                "cascade_probability": round(large_epidemics / num_trials, 4),
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
        """Assess how much each node's removal reduces epidemic spread.

        For every node *v* the method builds a subgraph *G − {v}*, runs
        *num_trials* SIR simulations on it, and compares the result to a
        baseline computed on the original graph with the same parameters.

        Metrics per node
        * **cascade_reduction** – baseline avg epidemic fraction minus the
          avg fraction with the node removed.  High = protecting this node
          is very effective.
        * **prob_reduction** – drop in large-epidemic probability.
        * **cascade_blocked** – avg epidemic fraction with node removed.
        * **prob_blocked** – large-epidemic probability with node removed.
        * **time_blocked** – avg rounds to stabilisation with node removed.
        * **degree / betweenness / closeness** – structural metrics on the
          original graph for reference.
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(1, int(seed_fraction * n))

        # ── Baseline on the full graph ──────────────────────────────────
        baseline_fractions: list[float] = []
        baseline_large = 0
        for _ in range(num_trials):
            s = set(random.sample(nodes, seed_size))
            res, _ = self.run(s)
            baseline_fractions.append(res.epidemic_fraction)
            if res.is_large_epidemic:
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

            sub_sim = SIRModel(sub, self.beta, self.gamma)
            sub_seed_size = max(1, min(seed_size, len(sub_nodes)))

            fracs: list[float] = []
            large = 0
            total_time = 0

            for _ in range(num_trials):
                s = set(random.sample(sub_nodes, sub_seed_size))
                res, _ = sub_sim.run(s)
                fracs.append(res.epidemic_fraction)
                total_time += res.time_to_epidemic
                if res.is_large_epidemic:
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
