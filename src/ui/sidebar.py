from __future__ import annotations

import streamlit as st

from simulation.seed_selection import SeedStrategy
from ui.state import SessionKeys, SidebarConfig, clear_sim_results


def render_sidebar(model: str = "bootstrap") -> SidebarConfig:
    graph = st.session_state.get(SessionKeys.GRAPH)

    # ── Active graph summary ───────────────────────────────────────────
    st.sidebar.header("Active Graph")
    if graph is not None:
        st.sidebar.info(
            f"**{graph.number_of_nodes()}** nodes · **{graph.number_of_edges()}** edges"
        )

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Change graph", use_container_width=True):
        clear_sim_results()
        st.session_state.pop(SessionKeys.GRAPH, None)
        st.rerun()
    if col_b.button("Change model", use_container_width=True):
        clear_sim_results()
        st.session_state.pop(SessionKeys.MODEL, None)
        st.rerun()

    st.sidebar.markdown("---")

    # ── Simulation parameters ──────────────────────────────────────────
    st.sidebar.header("Simulation Parameters")

    seed_fraction = st.sidebar.slider(
        "Initial infection fraction", 0.01, 1.0, 0.05, 0.01
    )
    num_trials = st.sidebar.number_input("Number of trials", 10, 500, 50, step=10)

    phi = 0.3
    if model == "bootstrap":
        threshold = st.sidebar.number_input("Bootstrap threshold (k)", 1, 50, 2)
        beta, gamma, switch_fraction = 0.3, 0.1, 0.2
    elif model == "sir":
        threshold = 2
        beta = st.sidebar.slider("Transmission rate (β)", 0.01, 1.0, 0.3, 0.01)
        gamma = st.sidebar.slider("Recovery rate (γ)", 0.01, 1.0, 0.1, 0.01)
        switch_fraction = 0.2
    elif model == "sis":
        threshold = 2
        beta = st.sidebar.slider("Transmission rate (β)", 0.01, 1.0, 0.3, 0.01)
        gamma = st.sidebar.slider("Recovery rate (μ)", 0.01, 1.0, 0.1, 0.01)
        switch_fraction = 0.2
    elif model == "wtm":
        threshold = 2
        beta, gamma, switch_fraction = 0.3, 0.1, 0.2
        phi = st.sidebar.slider(
            "Fractional threshold (φ)", 0.01, 1.0, 0.3, 0.01,
            help="A node activates when this fraction of its neighbours are infected.",
        )
    else:
        # Hybrid models
        _MODELS_WITH_THRESHOLD = {"h1", "h2"}
        if model in _MODELS_WITH_THRESHOLD:
            threshold = st.sidebar.number_input("Bootstrap threshold (k)", 1, 50, 2)
        else:
            threshold = 2

        _MODELS_WITH_PHI = {"h4", "h5", "h6"}
        if model in _MODELS_WITH_PHI:
            phi = st.sidebar.slider(
                "Fractional threshold (φ)", 0.01, 1.0, 0.3, 0.01,
                help="WTM threshold: a node activates when this fraction of its neighbours are infected.",
            )

        _MODELS_WITH_BETA = {"h1", "h2", "h3", "h4", "h5"}
        if model in _MODELS_WITH_BETA:
            beta = st.sidebar.slider("Transmission rate (β)", 0.01, 1.0, 0.3, 0.01)
        else:
            beta = 0.3

        gamma = st.sidebar.slider("Recovery rate (γ)", 0.01, 1.0, 0.1, 0.01)

        if model in {"h2", "h5"}:
            switch_fraction = st.sidebar.slider(
                "Switch threshold (f)", 0.01, 1.0, 0.2, 0.01,
                help="Fraction of the population that must be infected before the model switches phases.",
            )
        else:
            switch_fraction = 0.2

    _STRATEGY_LABELS = {
        "Random": SeedStrategy.RANDOM,
        "High Degree": SeedStrategy.HIGH_DEGREE,
        "High k-core": SeedStrategy.HIGH_KCORE,
    }
    strategy_label = st.sidebar.selectbox("Seeding strategy", list(_STRATEGY_LABELS))
    seed_strategy = _STRATEGY_LABELS[strategy_label]

    return SidebarConfig(
        threshold=int(threshold),
        seed_fraction=float(seed_fraction),
        num_trials=int(num_trials),
        beta=float(beta),
        gamma=float(gamma),
        seed_strategy=seed_strategy,
        switch_fraction=float(switch_fraction),
        phi=float(phi),
    )
