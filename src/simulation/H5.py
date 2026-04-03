"""
H5 — Sequential Hybrid: SIS then WTM.

Phase 1 — SIS mode: the network runs SIS dynamics (transmission rate β, recovery
rate γ).  The current infected fraction is monitored each round.

Switch condition: when the simultaneously infected fraction reaches switch_fraction f,
the model transitions to Phase 2.

Phase 2 — WTM mode: SIS transmission and recovery stop.  A susceptible node
activates if the fraction of ever-infected neighbours >= phi.  Ever-infected counts
both currently infected (I) and recovered (R from Phase 1), so recovered individuals
still signal danger and count toward a neighbour's WTM threshold.  Runs until stable.

If f is never reached, the model runs as pure SIS to termination.

Interpretation: early in an outbreak individuals respond to direct contacts (SIS
channel); once the epidemic becomes visibly widespread, threshold-driven social
contagion (WTM channel) takes over.

Parameters: β, γ (Phase 1), phi — WTM threshold (Phase 2), f — switch threshold.
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
class H5Result:
    """Stores the result of one H5 simulation run."""
    infected_nodes: set[int] = field(default_factory=set)   # all ever infected across both phases
    cascade_fraction: float = 0.0
    cascade_size: int = 0
    switched: bool = False              # whether the switch threshold was reached
    switch_size: int = 0               # simultaneously infected at switch moment
    switch_fraction: float = 0.0
    rounds_phase1: int = 0
    rounds_phase2: int = 0
    time_to_cascade: int = 0
    is_large_cascade: bool = False


@dataclass
class H5Metrics:
    """Stores the metrics collected across multiple H5 simulation trials."""
    cascade_size: float = 0.0
    critical_seed_size: int = 0
    cascade_probability: float = 0.0
    time_to_cascade: float = 0.0
    cascade_threshold: float = 0.0
    switch_probability: float = 0.0    # fraction of trials where switch was triggered
    avg_switch_fraction: float = 0.0   # avg simultaneous infected fraction at switch
    seed_strategy: str = SeedStrategy.RANDOM


class H5Model:
    """
    Sequential Hybrid: SIS then WTM (H5).

    Phase 1 — SIS:
      Each round, susceptible neighbours of infected nodes are infected with
      probability β; infected nodes recover with probability γ → back to susceptible.
      When the simultaneously infected fraction >= switch_fraction the model switches.

    Phase 2 — WTM:
      All ever-infected nodes (current I + recovered R from Phase 1) count toward
      the fractional threshold for susceptible nodes.  A susceptible node activates
      if infected_neighbours / degree >= phi.  Runs until stable.
    """

    def __init__(
        self,
        graph: nx.Graph,
        phi: float,
        beta: float,
        gamma: float,
        switch_fraction: float,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self.graph = graph
        self.phi = phi
        self.beta = beta
        self.gamma = gamma
        self.switch_fraction = switch_fraction
        self.max_steps = max_steps
        self.n = graph.number_of_nodes()

    def run(self, seed_nodes: set[int], record_sequence: bool = False) -> tuple[H5Result, list]:
        """
        Runs one H5 simulation starting from the given seed nodes.

        The activation_sequence uses (newly_infected, newly_recovered) tuples; Phase 2
        entries have an empty recovered set.

        Returns:
            result: H5Result
            activation_sequence: list of (set, set); empty if record_sequence is False.
        """
        infected = set(seed_nodes)
        susceptible = set(self.graph.nodes()) - infected
        recovered_phase1: set[int] = set()
        ever_infected = set(seed_nodes)

        rounds_phase1 = 0
        rounds_phase2 = 0
        switched = False
        switch_size = 0
        switch_frac = 0.0

        activation_sequence: list = []
        if record_sequence:
            activation_sequence.append((set(infected), set()))

        # ── Phase 1: SIS ────────────────────────────────────────────────
        while infected and rounds_phase1 < self.max_steps:
            if len(infected) / self.n >= self.switch_fraction:
                switched = True
                switch_size = len(infected)
                switch_frac = switch_size / self.n
                break

            newly_infected: set[int] = set()
            for node in infected:
                for neighbour in self.graph.neighbors(node):
                    if neighbour in susceptible and random.random() < self.beta:
                        newly_infected.add(neighbour)

            newly_recovered = {node for node in infected if random.random() < self.gamma}

            susceptible -= newly_infected
            susceptible |= newly_recovered
            infected -= newly_recovered
            infected |= newly_infected
            recovered_phase1 |= newly_recovered
            ever_infected |= newly_infected
            rounds_phase1 += 1

            if record_sequence and (newly_infected or newly_recovered):
                activation_sequence.append((set(newly_infected), set(newly_recovered)))

        # ── Phase 2: WTM ────────────────────────────────────────────────
        if switched:
            # active pool = all ever-infected (I + recovered from Phase 1)
            active = set(ever_infected)

            while True:
                newly_activated: set[int] = set()
                for node in susceptible:
                    deg = self.graph.degree(node)
                    if deg == 0:
                        continue
                    active_count = sum(1 for nb in self.graph.neighbors(node) if nb in active)
                    if active_count / deg >= self.phi:
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

        result = H5Result(
            infected_nodes=ever_infected,
            cascade_fraction=cascade_fraction,
            cascade_size=cascade_size,
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
            switch_prob: fraction of trials where switch was triggered.
            avg_switch_frac: average simultaneous infected fraction at switch.
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
        """Finds the minimum seed size for a large cascade using binary search."""
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
    ) -> H5Metrics:
        """Collects all H5 metrics for the given seed size."""
        critical_seed = self.find_critical_seed_size(num_trials=num_trials, seed=seed, strategy=strategy)
        cascade_thresh = critical_seed / self.n if self.n > 0 else 0.0

        prob, avg_fraction, avg_time, switch_prob, avg_switch_frac = self.cascade_probability(
            seed_size=seed_size, num_trials=num_trials, seed=seed, strategy=strategy
        )

        return H5Metrics(
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
        """Compute per-node influence on cascade spread under H5 dynamics."""
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
        """Assess how much each node's removal reduces cascade spread under H5 dynamics."""
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

            sub_sim = H5Model(sub, self.phi, self.beta, self.gamma, self.switch_fraction, self.max_steps)
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
