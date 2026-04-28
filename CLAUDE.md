# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Hybrid SIR / Bootstrap Percolation — Network Contagion Research

A research codebase studying hybrid contagion models that combine SIR epidemic dynamics with threshold-based spreading (bootstrap percolation / Watts Threshold Model) on networks. The goal is to map phase diagrams and identify regimes where hybrid models behave qualitatively differently from either mechanism alone.

**Core hybrid mechanism (H1):** a susceptible node becomes infected if either (1) an infected neighbour transmits at rate β (SIR channel), or (2) it has ≥ k infected neighbours simultaneously (bootstrap channel). Infected nodes recover at rate γ. Limits: k → ∞ reduces to pure SIR; β → 0 reduces to pure bootstrap percolation.

Two interfaces: a CLI (`src/Main.py`) and a Streamlit web UI (`src/app.py`).

## Tech Stack

- **Python 3.13+**
- **networkx** — graph data structures and centrality metrics
- **streamlit** — web UI framework
- **plotly** — interactive charts and cascade animations
- **matplotlib**, **pandas**, **statsmodels** — analysis and plotting

Dependencies: `src/requirements.txt`

## Run

```bash
# Web UI
streamlit run src/app2.py
# or
./run.sh

# CLI
python src/Main.py

# Standalone experiment scripts (save CSV + PNG before returning)
python src/experiments/bootstrap_analysis.py
python src/experiments/h1_analysis.py
# Results land in src/experiments/results/{model}/
```

Install dependencies: `pip install -r src/requirements.txt`

## Implemented Models

Ten contagion models share a common interface (see *Adding a New Model* below):

| ID | Class | Mechanism |
|---|---|---|
| Bootstrap | `BootstrapPercolation` | Deterministic threshold k on infected-neighbour count |
| SIR | `SIRModel` | Probabilistic β per edge, γ recovery |
| SIS | `SISModel` | Like SIR but recovered → susceptible (endemic equilibria) |
| WTM | `WTMModel` | Fractional threshold φ (Watts Threshold Model) |
| H1 | `H1Model` | OR-fusion: SIR OR bootstrap (either channel fires) |
| H2 | `H2Model` | Sequential: SIR phase until fraction f ever-infected, then bootstrap |
| H3 | `H3Model` | Soft bootstrap: probabilistic infection proportional to infected-neighbour count |
| H4 | `H4Model` | OR-fusion: SIS OR WTM |
| H5 | `H5Model` | Sequential: SIS phase, then WTM |
| H6 | `H6Model` | Soft WTM: probabilistic fractional-threshold infection |

## Key Directories

| Path | Purpose |
|---|---|
| `src/simulation/` | All contagion models (bootstrap.py, sir.py, sis.py, wtm.py, H1–H6.py) and seed selection |
| `src/input/` | Graph generation (ER, geometric, lattice, scale-free) and file loading (DIMACS, edge list, GML) |
| `src/analysis/` | Graph structural metrics, model-specific parameter sweeps, phase diagram mapping |
| `src/visualization/` | Plotly-based cascade animation and phase diagram plots |
| `src/ui/` | Streamlit app — sidebar config, session state, and 5 tab types × 10 models = ~50 tab files |
| `src/experiments/` | Standalone experiment scripts; results saved to `src/experiments/results/{model}/` |
| `data/` | Sample graph files |

## Critical Files

- `src/simulation/bootstrap.py:40` — `BootstrapPercolation` class; the central algorithm and reference implementation for all models
- `src/simulation/bootstrap.py:16` — `BootstrapResult` and `PercolationMetrics` dataclasses; canonical data contract pattern
- `src/ui/state.py:1` — `SessionKeys` enum and `SidebarConfig` dataclass; how UI state is shared across all tabs
- `src/ui/sidebar.py:1` — graph loading/generation + parameter controls rendered on every page
- `src/ui/app_entry.py:27` — tab layout definition; where new models must be registered
- `src/input/graph_loader.py:1` — supported file formats and parsing logic
- `src/analysis/parameter_sweep.py:1` — generic sweep runner; model-specific sweeps delegate to this

## Coding Conventions

- Keep simulation, analysis, and plotting in separate modules — never mix them
- Every experiment must fix and log all random seeds for reproducibility
- Always save results as CSV or JSON before plotting; never generate a plot without persisting the underlying data first
- Keep code simple. Prioritize ease of understanding over complex lines if they do the same function

### Parameter naming

| Symbol | Variable name | Meaning |
|---|---|---|
| k | `threshold` | Bootstrap percolation threshold (integer) |
| β | `beta` | Transmission rate per edge, [0,1] |
| γ | `gamma` | Recovery rate per node, [0,1] |
| φ | `phi` | Fractional threshold for WTM, [0,1] |
| ρ | `seed_fraction` | Initial infection fraction, [0,1] |
| f | `switch_fraction` | Phase-switch fraction for H2/H5, [0,1] |

### Model interface

Every model class exposes the same methods:

```python
class XxxModel:
    def __init__(self, graph, ...): ...
    def run(self, seed_nodes, record_sequence=False) -> (Result, list): ...
    def cascade_probability(self, seed_size, num_trials) -> (prob, avg_frac, avg_time): ...
    def collect_metrics(self, seed_size, num_trials, seed, strategy) -> Metrics: ...
    def node_influence_analysis(...) -> list[dict]: ...
    def node_blocking_analysis(...) -> (list[dict], baseline_avg, baseline_prob): ...
```

Cascade-based models name their result fields `cascade_*`; epidemic models use `epidemic_*`.

## UI Session State Pattern

`SessionKeys` (in `src/ui/state.py`) is the single source of truth for all inter-tab communication. Every model needs its own set of keys:

```python
# state.py pattern for a new model Xxx
XXX_SIM_RESULTS = "xxx_sim_results"    # dict: result, metrics, seed_size, n
XXX_VULN_DATA   = "xxx_vuln_data"      # list[dict] from node_influence_analysis
XXX_BLOCK_DATA  = "xxx_block_data"     # list[dict] from node_blocking_analysis
```

Tab renderers read/write these keys; they never call each other directly.

## Adding a New Model

To add model `Hx`:

1. `src/simulation/Hx.py` — implement the model class following the interface above, plus `HxResult` and `HxMetrics` dataclasses
2. `src/analysis/hx_parameter_sweep.py` — model-specific sweep functions
3. `src/ui/tabs/hx_{simulation,animation,vulnerability,sweep}_tab.py` — four tab renderers (`render_hx_*_tab(graph, config)`)
4. `src/ui/state.py` — add `HX_SIM_RESULTS`, `HX_VULN_DATA`, `HX_BLOCK_DATA` to `SessionKeys`; add Hx fields to `SidebarConfig` if new parameters are needed
5. `src/ui/sidebar.py` — add parameter sliders for any new parameters, conditioned on `model == "Hx"`
6. `src/ui/app_entry.py` — register the model in both `_render_welcome()` (button) and the tab dispatch
7. `src/experiments/hx_analysis.py` — standalone experiment script

## Additional Docs

- [Simulation Algorithm](.claude/docs/simulation.md) — hybrid model definition, bootstrap percolation implementation, SIR integration, metrics
- [UI Architecture](.claude/docs/ui.md) — Streamlit tab structure, session state, sidebar config flow
