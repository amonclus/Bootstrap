# UI Architecture

## Entry Points

- `src/app.py` — Streamlit entry point, calls `run_app()`
- `src/ui/app_entry.py:14` — `run_app()` sets page config, renders sidebar, then renders 5 tabs
- `src/Main.py` — CLI interface with interactive menu (generate graph → set params → run analysis)

## Session State

`src/ui/state.py` manages all shared state:

- `SessionKeys` enum (`state.py:1`) — typed keys for `st.session_state`; avoids magic strings across tabs
- `SidebarConfig` frozen dataclass (`state.py:17`) — holds `threshold`, `seed_fraction`, `num_trials`; passed to every tab render function
- `get_graph_or_stop()` (`state.py:30`) — retrieves the current graph or calls `st.stop()` if none loaded; every tab calls this to guard against rendering without a graph

## Sidebar

`src/ui/sidebar.py` — rendered on every page load. Handles:
1. Graph source selection: generate (ER / geometric / lattice) or upload file
2. Parameter controls: threshold, seed fraction, number of trials
3. Stores graph and config into session state

## Tabs

All tab renderers live in `src/ui/tabs/` and share the same signature: `render_*(graph, config)`.

| Tab | File | What it does |
|---|---|---|
| Graph Statistics | `tabs/stats_tab.py` | Calls `compute_graph_statistics()`, displays structural metrics |
| Cascade Simulation | `tabs/simulation_tab.py` | Runs `collect_metrics()`, shows averaged results |
| Cascade Animation | `tabs/animation_tab.py` | Runs single cascade with history, renders Plotly animation |
| Node Vulnerability | `tabs/vulnerability_tab.py` | Runs influence + blocking analysis, renders ranked node tables |
| Parameter Sweep | `tabs/sweep_tab.py` | Calls `parameter_sweep.py` functions, plots heatmaps |

## Visualization

`src/visualization/visualization.py` — builds a Plotly figure with one frame per cascade round. Detects graph type (lattice vs. general) to choose layout. Edge opacity scales with graph density.
