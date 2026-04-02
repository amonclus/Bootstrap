"""
Algorithm module responsible for implementing the bootstrap percolation algorithm and collecting all experimental metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds


@dataclass
class BootstrapResult:
    """
    Stores the result of the bootstrap percolation algorithm.
    """
    infected_nodes: set[int] = field(default_factory=set)
    cascade_size: int = 0
    cascade_fraction: float = 0.0
    time_to_cascade: int = 0
    is_full_cascade: bool = False


@dataclass
class PercolationMetrics:
    """
    Stores the metrics collected during the percolation algorithm.
    """
    cascade_size: float = 0.0  # average cascade fraction (infected / total nodes)
    critical_seed_size: int = 0  # minimum seeds for full cascade
    cascade_probability: float = 0.0  # probability of full cascade
    time_to_cascade: float = 0.0  # average rounds to stabilize
    percolation_threshold: float = 0.0  # critical seed fraction
    seed_strategy: str = SeedStrategy.RANDOM  # strategy used to select initial seeds


class BootstrapPercolation:
    """
    Class responsible for implementing the bootstrap percolation algorithm.
    """
    def __init__(self, graph: nx.Graph, threshold: int = 2) -> None:
        self.graph = graph
        self.threshold = threshold
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[BootstrapResult, list[set]]:
        """
        Runs the main bootstrap percolation algorithm starting from the given seed nodes.
        Args:
            seed_nodes: set[int]
                Initial set of infected nodes (the "seed" of the cascade).
                The algorithm will start with these nodes infected and then
                iteratively infect new nodes based on the threshold condition.
            record_sequence: bool
                If True, the method will record each step of the percolation algorithm for later display.

        Returns:
            result : BootstrapResult

        """
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

    def cascade_probability(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> tuple[float, float, float]:
        """
        Measures the cascade probability of the studied graph.
        Args:
            seed_size: int
                Size of the set of initially infected nodes.
            num_trials:
                Number of independent trials to run for estimating the probability.
            seed:
                Optional random seed for reproducibility. If provided, it will be used to seed the random number generator before selecting seed nodes in each trial.
            strategy:
                Seed-selection strategy to use. One of SeedStrategy.RANDOM, HIGH_DEGREE, HIGH_KCORE.

        Returns:
            prob: int
                Probability of cascade.
            avg_fraction: float
                Average fraction of infected nodes across all trials.
            avg_time: float
                Average number of rounds until cascade stabilisation across all trials.
        """
        if seed is not None:
            random.seed(seed)

        full_cascades = 0
        total_fraction = 0.0
        total_time = 0

        for _ in range(num_trials):
            seed_nodes = select_seeds(self.graph, seed_size, strategy)
            result, _ = self.run(seed_nodes)
            total_fraction += result.cascade_fraction
            total_time += result.time_to_cascade
            if result.is_full_cascade:
                full_cascades += 1

        prob = full_cascades / num_trials
        avg_fraction = total_fraction / num_trials
        avg_time = total_time / num_trials
        return prob, avg_fraction, avg_time

    def find_critical_seed_size(self, num_trials: int = 50, cascade_threshold: float = 1.0, probability_threshold: float = 0.5, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> int:
        """
        Finds the critical seed size based on the threshold condition.
        Args:
            num_trials: int
                Number of independent trials to run for estimating the probability at each seed size.
            cascade_threshold: float
                    Fraction of nodes that must be infected for a trial to be considered a "full cascade". Default is 1.0 (i.e., all nodes must be infected).
            probability_threshold: float
                The minimum probability of achieving a full cascade required to consider a seed size as critical. Default is 0.5 (i.e., at least 50% of trials must result in a full cascade).
            seed: int
                Optional random seed for reproducibility. If provided, it will be used to seed the random number generator before selecting seed nodes in each trial.
        Returns:
            result: int
                The critical seed size, defined as the smallest number of initially infected nodes required to achieve a full cascade with at least the specified probability threshold.
        """
        if seed is not None:
            random.seed(seed)

        low, high = 1, self.n
        result = self.n  # worst case: all nodes

        while low <= high:
            mid = (low + high) // 2
            prob, _, _ = self.cascade_probability(mid, num_trials, strategy=strategy)
            if prob >= probability_threshold:
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return result

    def find_percolation_threshold(self, num_trials: int = 50, probability_threshold: float = 0.5, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> float:
        """
        Finds the percolation threshold based on the threshold condition.
        Args:
            num_trials: int
                Number of independent trials to run for estimating the probability at each seed size.
            probability_threshold: float
                The minimum probability of achieving a full cascade required to consider a seed size as critical. Default is 0.5 (i.e., at least 50% of trials must result in a full cascade).
            seed: int
                Optional random seed for reproducibility. If provided, it will be used to seed the random number generator before selecting seed nodes in each trial.

        Returns:
            result: float
                The percolation threshold, defined as the critical seed size divided by the total number of nodes in the graph. This represents the minimum fraction of initially infected nodes required to achieve a full cascade with at least the specified probability threshold.
        """
        critical = self.find_critical_seed_size(
            num_trials=num_trials,
            probability_threshold=probability_threshold,
            seed=seed,
            strategy=strategy,
        )
        result = critical / self.n if self.n > 0 else 0.0
        return result

    def collect_metrics(self, seed_size: int, num_trials: int = 100, seed: Optional[int] = None, strategy: SeedStrategy | str = SeedStrategy.RANDOM) -> PercolationMetrics:
        """
        Collects all metrics related to the percolation process, including cascade size, critical seed size, cascade probability, time to cascade, and percolation threshold.
        Args:
            seed_size: int
                Size of the set of initially infected nodes for evaluating cascade probability and average cascade size.
            num_trials: int
                Number of independent trials to run for estimating the probability at each seed size.
            seed: int
                Optional random seed for reproducibility. If provided, it will be used to seed the random number generator before selecting seed nodes in each trial.

        Returns:
            result: PercolationMetrics
                An object containing all collected metrics related to the percolation process.
        """
        # Find critical seed size and percolation threshold
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)

        percolation_thresh = critical_seed / self.n if self.n > 0 else 0.0

        # Estimate cascade probability and averages at the given seed size
        prob, avg_fraction, avg_time = self.cascade_probability(seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy)

        result = PercolationMetrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            percolation_threshold=percolation_thresh,
            seed_strategy=SeedStrategy(strategy).value,
        )

        return result

    # ── Node vulnerability analysis ─────────────────────────────────────

    def node_influence_analysis(self, seed_fraction: float = 0.05,num_trials: int = 20,seed: Optional[int] = None,progress_callback=None,) -> list[dict]:
        """Compute per-node influence on cascade propagation.

        For every node *v* in the graph the method runs *num_trials*
        bootstrap-percolation simulations where *v* is **always** part of
        the initial seed set (the remaining seeds are chosen uniformly at
        random).

        Metrics collected per node
        * **influence_score** – average cascade fraction across trials.
        * **cascade_probability** – fraction of trials that produced a
          full cascade (all nodes infected).
        * **avg_time** – average number of rounds until stabilisation.
        * **cascade_std** – standard deviation of the cascade fraction
          (high → unpredictable, low → consistent behaviour).

        A high influence score *and* high cascade probability means the
        node is a **weak point**.  A low influence score means it is
        **strong** / resilient.
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(2, int(seed_fraction * n))

        # Pre-compute structural centralities (cheap for typical sizes)
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

        # Sort: most influential (weakest) first
        results.sort(key=lambda r: r["influence_score"], reverse=True)
        return results

    # ── Node blocking / immunisation analysis ───────────────────────────

    def node_blocking_analysis(self,seed_fraction: float = 0.05,num_trials: int = 20,seed: Optional[int] = None,progress_callback=None,) -> tuple[list[dict], float, float]:
        """Assess how much each node's removal reduces cascade spread.

        For every node *v* the method builds a subgraph *G − {v}*,
        runs *num_trials* bootstrap-percolation simulations on it, and
        compares the result to a **baseline** computed on the original
        graph with the same seed fraction and trial count.

        Metrics per node
        * **cascade_reduction** – baseline avg cascade fraction minus
          the avg fraction with the node removed.  High = protecting
          this node is very effective.
        * **prob_reduction** – drop in full-cascade probability.
        * **cascade_blocked** – avg cascade fraction with node removed.
        * **prob_blocked** – full-cascade probability with node removed.
        * **time_blocked** – avg rounds to stabilisation with node
          removed.
        * **degree / betweenness / closeness** – structural metrics on
          the *original* graph for reference.
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        n = len(nodes)
        seed_size = max(1, int(seed_fraction * n))

        # ── Baseline on the full graph ──────────────────────────────────
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

        # Pre-compute structural centralities on original graph
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)

        results: list[dict] = []

        for idx, target in enumerate(nodes):
            # Build reduced graph
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

            sub_sim = BootstrapPercolation(sub, self.threshold)
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

