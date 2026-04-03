from __future__ import annotations

import os
import tempfile

import streamlit as st

from input.graph_generator import (
    generate_er_graph,
    generate_lattice_graph,
    generate_random_geometric_graph,
)
from input.graph_loader import (
    load_graph_from_dimacs,
    load_graph_from_edge_list,
    load_graph_from_gml,
)
from simulation.seed_selection import SeedStrategy
from ui.state import SessionKeys, SidebarConfig


def render_sidebar(model: str = "bootstrap") -> SidebarConfig:
    st.sidebar.header("Graph Configuration")

    source = st.sidebar.radio("Graph source", ["Generate", "Upload file"])

    if source == "Generate":
        _render_generate_graph_controls()
    else:
        _render_upload_graph_controls()

    st.sidebar.markdown("---")
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
        # Hybrid models — show only the parameters each model uses
        _MODELS_WITH_THRESHOLD = {"h1", "h2"}
        if model in _MODELS_WITH_THRESHOLD:
            threshold = st.sidebar.number_input("Bootstrap threshold (k)", 1, 50, 2)
        else:
            threshold = 2  # unused default for models without a hard threshold

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
                "Switch threshold (f)",
                0.01, 1.0, 0.2, 0.01,
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

    st.sidebar.markdown("---")
    if st.sidebar.button("↩ Change model"):
        st.session_state.pop(SessionKeys.MODEL, None)
        st.session_state.pop(SessionKeys.GRAPH, None)
        st.rerun()

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


def _render_generate_graph_controls() -> None:
    graph_type = st.sidebar.selectbox(
        "Graph type", ["Erdős–Rényi", "Random Geometric", "Lattice"]
    )

    if graph_type == "Erdős–Rényi":
        n = st.sidebar.number_input("Number of nodes", 10, 5000, 100)
        p = st.sidebar.slider("Edge probability (p)", 0.01, 1.0, 0.1, 0.01)
        if not st.sidebar.button("Generate graph"):
            return

        with st.spinner("Generating graph…"):
            graph = generate_er_graph(int(n), float(p))
    elif graph_type == "Random Geometric":
        n = st.sidebar.number_input("Number of nodes", 10, 5000, 100)
        radius = st.sidebar.slider("Connection radius (r)", 0.01, 1.0, 0.2, 0.01)
        if not st.sidebar.button("Generate graph"):
            return

        with st.spinner("Generating graph…"):
            graph = generate_random_geometric_graph(int(n), float(radius))
    else:
        grid_size = st.sidebar.number_input("Grid side length", 3, 100, 10)
        if not st.sidebar.button("Generate graph"):
            return

        with st.spinner("Generating graph…"):
            graph = generate_lattice_graph(int(grid_size))

    st.session_state[SessionKeys.GRAPH] = graph
    st.session_state.pop(SessionKeys.SIM_RESULTS, None)
    st.sidebar.success(
        f"Graph created – {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )


def _render_upload_graph_controls() -> None:
    fmt = st.sidebar.selectbox("File format", ["DIMACS", "Edge List", "GML"])
    uploaded = st.sidebar.file_uploader(
        "Upload graph file", type=["dimacs", "txt", "gml", "edgelist"]
    )

    if uploaded is None:
        return

    suffix = {"DIMACS": ".dimacs", "Edge List": ".txt", "GML": ".gml"}[fmt]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        if fmt == "DIMACS":
            graph = load_graph_from_dimacs(tmp_path)
        elif fmt == "Edge List":
            graph = load_graph_from_edge_list(tmp_path)
        else:
            graph = load_graph_from_gml(tmp_path)

        st.session_state[SessionKeys.GRAPH] = graph
        st.session_state.pop(SessionKeys.SIM_RESULTS, None)
        st.sidebar.success(
            f"Graph loaded – {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
    finally:
        os.unlink(tmp_path)

