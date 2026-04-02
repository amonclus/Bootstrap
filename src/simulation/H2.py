"""
H2 — Sequential Hybrid (Switching Model).

Phase 1 — SIR mode: the network runs standard SIR dynamics (transmission rate β,
recovery rate γ).  The fraction of ever-infected nodes is monitored each round.

Switch condition: when the ever-infected fraction reaches the switch threshold f,
the model transitions to Phase 2.

Phase 2 — Bootstrap mode: SIR transmission and recovery stop.  A susceptible
node is now infected if it has ≥ threshold infected neighbours, where "infected"
counts ALL ever-infected nodes (both currently infected I and recovered R).
Recovered individuals remain "aware" and their presence still signals danger,
so they continue to count toward a susceptible neighbour's threshold.  The
bootstrap phase runs until no new nodes can be activated.

If the switch threshold f is never reached the model runs as pure SIR to
termination.

Interpretation: early in an outbreak individuals respond to direct contacts
(SIR channel); once the epidemic becomes visibly widespread, fear-driven social
reinforcement (bootstrap channel) takes over.

Parameters: β, γ (Phase 1), k — bootstrap threshold (Phase 2), f — switch
threshold (fraction of population that triggers the regime change).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from simulation.seed_selection import SeedStrategy, select_seeds

_LARGE_CASCADE_FRACTION = 0.5


@dataclass
class H2Result:
    """Stores the result of one H2 simulation run."""
    infected_nodes: set[int] = field(default_factory=set)   # all ever infected (I ∪ R across both phases)
    recovered_nodes: set[int] = field(default_factory=set)  # nodes that recovered during Phase 1
    cascade_size: int = 0
    cascade_fraction: float = 0.0
    switched: bool = False              # whether the switch threshold was reached
    switch_size: int = 0               # ever-infected count at the moment of switching
    switch_fraction: float = 0.0      # ever-infected fraction at the moment of switching
    rounds_phase1: int = 0            # SIR rounds completed before switch (or total if no switch)
    rounds_phase2: int = 0            # bootstrap rounds after switch
    time_to_cascade: int = 0          # total rounds (phase1 + phase2)
    is_large_cascade: bool = False


@dataclass
class H2Metrics:
    """Stores the metrics collected across multiple H2 simulation trials."""
    cascade_size: float = 0.0          # average cascade fraction across trials
    critical_seed_size: int = 0        # minimum seeds for a large cascade
    cascade_probability: float = 0.0   # fraction of trials that produced a large cascade
    time_to_cascade: float = 0.0       # average total rounds to stabilise
    cascade_threshold: float = 0.0     # critical_seed_size / n
    switch_probability: float = 0.0    # fraction of trials where the switch was triggered
    avg_switch_fraction: float = 0.0   # average ever-infected fraction when switch occurred
    seed_strategy: str = SeedStrategy.RANDOM


class H2Model:
    """
    Sequential Hybrid contagion model (H2).

    Phase 1 — SIR:
      Each round, susceptible neighbours of infected nodes are infected with
      probability β; infected nodes recover with probability γ.  The ever-infected
      fraction is tracked.  When it reaches switch_fraction the model switches.

    Phase 2 — Bootstrap:
      All ever-infected nodes (I ∪ R) count toward the threshold for susceptible
      nodes.  A susceptible node is infected if it has ≥ threshold such neighbours.
      No further SIR transmission or recovery occurs.  Runs until stable.
    """

    def __init__(
        self,
        graph: nx.Graph,
        threshold: int,
        beta: float,
        gamma: float,
        switch_fraction: float,
    ) -> None:
        self.graph = graph
        self.threshold = threshold
        self.beta = beta
        self.gamma = gamma
        self.switch_fraction = switch_fraction
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[H2Result, list]:
        """
        Runs one H2 simulation starting from the given seed nodes.

        The activation_sequence uses the same (newly_infected, newly_recovered)
        tuple format as the SIR model, compatible with animate_cascade.
        Phase 2 entries have an empty recovered set.

        Returns:
            result: H2Result
            activation_sequence: list of (set, set); empty if record_sequence is False.
        """
        infected = set(seed_nodes)
        susceptible = set(self.graph.nodes()) - infected
        recovered: set[int] = set()
        ever_infected = set(seed_nodes)

        rounds_phase1 = 0
        rounds_phase2 = 0
        switched = False
        switch_size = 0
        switch_frac = 0.0

        activation_sequence: list = []
        if record_sequence:
            activation_sequence.append((set(infected), set()))

        # ── Phase 1: SIR ────────────────────────────────────────────────
        while infected:
            # Check switch condition before this round
            if len(ever_infected) / self.n >= self.switch_fraction:
                switched = True
                switch_size = len(ever_infected)
                switch_frac = switch_size / self.n
                break

            newly_infected: set[int] = set()
            for node in infected:
                for neighbour in self.graph.neighbors(node):
                    if neighbour in susceptible and random.random() < self.beta:
                        newly_infected.add(neighbour)

            newly_recovered = {node for node in infected if random.random() < self.gamma}

            susceptible -= newly_infected
            infected -= newly_recovered
            infected |= newly_infected
            recovered |= newly_recovered
            ever_infected |= newly_infected
            rounds_phase1 += 1

            if record_sequence and (newly_infected or newly_recovered):
                activation_sequence.append((set(newly_infected), set(newly_recovered)))

        # ── Phase 2: Bootstrap ──────────────────────────────────────────
        if switched:
            # Active pool for threshold counting = all ever-infected (I ∪ R)
            active = set(ever_infected)

            while True:
                newly_activated: set[int] = set()
                for node in susceptible:
                    active_neighbours = sum(
                        1 for nb in self.graph.neighbors(node) if nb in active
                    )
                    if active_neighbours >= self.threshold:
                        newly_activated.add(node)

                if not newly_activated:
                    break

                susceptible -= newly_activated
                active |= newly_activated
                ever_infected |= newly_activated
                rounds_phase2 += 1

                if record_sequence:
                    activation_sequence.append((set(newly_activated), set()))

        cascade_size = len(ever_infected)
        cascade_fraction = cascade_size / self.n if self.n > 0 else 0.0

        result = H2Result(
            infected_nodes=ever_infected,
            recovered_nodes=recovered,
            cascade_size=cascade_size,
            cascade_fraction=cascade_fraction,
            switched=switched,
            switch_size=switch_size,
            switch_fraction=switch_frac,
            rounds_phase1=rounds_phase1,
            rounds_phase2=rounds_phase2,
            time_to_cascade=rounds_phase1 + rounds_phase2,
            is_large_cascade=(cascade_fraction >= _LARGE_CASCADE_FRACTION),
        )

        return result, activation_sequence if record_sequence else []

    def cascade_probability(
        self,
        seed_size: int,
        num_trials: int = 100,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> tuple[float, float, float, float, float]:
        """
        Estimates cascade statistics for a given seed size.

        Returns:
            prob: fraction of trials with a large cascade.
            avg_fraction: average cascade fraction.
            avg_time: average total rounds.
            switch_prob: fraction of trials where the switch was triggered.
            avg_switch_frac: average ever-infected fraction at switch (over switched trials only,
                             or 0.0 if no trial switched).
        """
        if seed is not None:
            random.seed(seed)

        large_cascades = 0
        total_fraction = 0.0
        total_time = 0
        switched_count = 0
        total_switch_frac = 0.0

        for _ in range(num_trials):
            seed_nodes = set(select_seeds(self.graph, seed_size, strategy))
            result, _ = self.run(seed_nodes)
            total_fraction += result.cascade_fraction
            total_time += result.time_to_cascade
            if result.is_large_cascade:
                large_cascades += 1
            if result.switched:
                switched_count += 1
                total_switch_frac += result.switch_fraction

        avg_switch_frac = total_switch_frac / switched_count if switched_count > 0 else 0.0

        return (
            large_cascades / num_trials,
            total_fraction / num_trials,
            total_time / num_trials,
            switched_count / num_trials,
            avg_switch_frac,
        )

    def find_critical_seed_size(
        self,
        num_trials: int = 50,
        probability_threshold: float = 0.5,
        seed: Optional[int] = None,
        strategy: SeedStrategy | str = SeedStrategy.RANDOM,
    ) -> int:
        """Finds the minimum seed size for a large cascade with binary search."""
        if seed is not None:
            random.seed(seed)

        low, high = 1, self.n
        result = self.n

        while low <= high:
            mid = (low + high) // 2
            prob, _, _, _, _ = self.cascade_probability(mid, num_trials, strategy=strategy)
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
    ) -> H2Metrics:
        """Collects all H2 metrics for the given seed size."""
        critical_seed = self.find_critical_seed_size(
            num_trials=num_trials, seed=seed, strategy=strategy
        )
        cascade_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time, switch_prob, avg_switch_frac = self.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return H2Metrics(
            cascade_size=avg_fraction,
            critical_seed_size=critical_seed,
            cascade_probability=prob,
            time_to_cascade=avg_time,
            cascade_threshold=cascade_thresh,
            switch_probability=switch_prob,
            avg_switch_fraction=avg_switch_frac,
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
        """Compute per-node influence on cascade spread under H2 dynamics."""
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
            switched_count = 0
            total_time = 0

            for _ in range(num_trials):
                companions = set(random.sample(other_nodes, pick))
                seed_set = companions | {target}
                result, _ = self.run(seed_set)
                fractions.append(result.cascade_fraction)
                total_time += result.time_to_cascade
                if result.is_large_cascade:
                    large_cascades += 1
                if result.switched:
                    switched_count += 1

            avg_fraction = sum(fractions) / num_trials
            mean_sq = sum(f * f for f in fractions) / num_trials
            std_fraction = (mean_sq - avg_fraction ** 2) ** 0.5

            results.append({
                "node": target,
                "influence_score": round(avg_fraction, 6),
                "cascade_probability": round(large_cascades / num_trials, 4),
                "switch_probability": round(switched_count / num_trials, 4),
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
        """Assess how much each node's removal reduces cascade spread under H2 dynamics."""
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

            sub_sim = H2Model(sub, self.threshold, self.beta, self.gamma, self.switch_fraction)
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
