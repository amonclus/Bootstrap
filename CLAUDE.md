# Hybrid SIR / Bootstrap Percolation — Network Contagion Research

A research codebase studying a hybrid contagion model that runs SIR epidemic dynamics and bootstrap percolation simultaneously on a network. The goal is to map the phase diagram in (β/γ, k) space and identify regimes where the hybrid model behaves qualitatively differently from either mechanism alone.

**Model:** a susceptible node becomes infected if either (1) an infected neighbour transmits at rate β (SIR channel), or (2) it has ≥ k infected neighbours simultaneously (bootstrap channel). Infected nodes recover at rate γ regardless of which channel fired. Limits: k → ∞ reduces to pure SIR; β → 0 reduces to pure bootstrap percolation.

**Key parameters:** β (transmission rate), γ (recovery rate), k (bootstrap threshold), network type (lattice, Erdős–Rényi, Barabási–Albert).

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
streamlit run src/app.py
# or
./run.sh

# CLI
python src/Main.py
```

Install dependencies: `pip install -r src/requirements.txt`

## Coding Conventions

- Keep simulation, analysis, and plotting in separate modules — never mix them
- Every experiment must fix and log all random seeds for reproducibility
- Always save results as CSV or JSON before plotting; never generate a plot without persisting the underlying data first
- Keep code simple. Prioritize ease of understanding over complex lines if they do the same function

## Key Directories

| Path | Purpose |
|---|---|
| `src/simulation/` | Hybrid model, bootstrap percolation algorithm, seed selection |
| `src/input/` | Graph generation (ER, geometric, lattice, scale-free) and file loading (DIMACS, edge list, GML) |
| `src/analysis/` | Graph structural metrics, parameter sweeps, phase diagram mapping |
| `src/visualization/` | Plotly-based cascade animation and phase diagram plots |
| `src/ui/` | Streamlit app — sidebar config, session state, and 5 tabs |
| `data/` | Sample graph files and persisted experiment results (CSV/JSON) |

## Critical Files

- `src/simulation/bootstrap.py:40` — `BootstrapPercolation` class; the central algorithm
- `src/simulation/bootstrap.py:16` — `BootstrapResult` and `PercolationMetrics` dataclasses; the data contract between simulation and UI
- `src/ui/state.py:1` — `SessionKeys` enum and `SidebarConfig` dataclass; how UI state is shared across tabs
- `src/ui/sidebar.py:1` — graph loading/generation + parameter controls rendered on every page
- `src/ui/app_entry.py:27` — tab layout definition (Stats, Simulation, Animation, Vulnerability, Sweep)
- `src/input/graph_loader.py:1` — supported file formats and parsing logic
- `src/analysis/parameter_sweep.py:1` — experiment runner for sweeping threshold/seed-fraction space

## Additional Docs

- [Simulation Algorithm](.claude/docs/simulation.md) — hybrid model definition, bootstrap percolation implementation, SIR integration, metrics
- [UI Architecture](.claude/docs/ui.md) — Streamlit tab structure, session state, sidebar config flow
