# Simulation Algorithm

## Hybrid Model

A susceptible node becomes infected if either channel fires:
1. **SIR channel** — an infected neighbour transmits at per-edge rate β; infected nodes recover at rate γ
2. **Bootstrap channel** — the node has ≥ k infected neighbours simultaneously (deterministic, immediate)

Special cases: k → ∞ = pure SIR; β → 0 = pure bootstrap percolation.

## Bootstrap Percolation (existing)

The core algorithm is in `src/simulation/bootstrap.py`. The `BootstrapPercolation` class wraps a `networkx` graph and a threshold value.

**Algorithm** (`bootstrap.py:71`): Each round, activate any uninfected node whose infected neighbor count ≥ threshold. Repeat until no new nodes activate. This is deterministic given a seed set.

**Entry point** (`bootstrap.py:49`): `run(seed_nodes, track_history)` — runs one cascade and returns a `BootstrapResult`.

## Metrics

`collect_metrics(seed_fraction, num_trials)` (`bootstrap.py:197`) runs multiple trials and returns a `PercolationMetrics` with:

- `cascade_size` — average fraction of nodes infected
- `critical_seed_size` — minimum seeds needed to trigger full cascade (binary search, `bootstrap.py:141`)
- `cascade_probability` — fraction of trials that produce a full cascade
- `time_to_cascade` — average rounds to stabilize
- `percolation_threshold` — critical seed size / total nodes

## Node Analysis

Both methods in `bootstrap.py` return per-node dicts that include structural properties (degree, betweenness, closeness).

- **Influence** (`bootstrap.py:235`): Forces each node into the seed set across many trials, measures average cascade fraction. Identifies which nodes trigger the largest cascades.
- **Blocking** (`bootstrap.py:309`): Removes each node and re-runs on the induced subgraph. Measures cascade reduction = baseline − blocked. Identifies nodes critical to protect.

## Seed Selection

`src/simulation/seed_selection.py` — `random_seeds(graph, fraction)` picks a random subset of nodes as the initial infected set.
