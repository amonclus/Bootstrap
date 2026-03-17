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
from ui.state import SessionKeys, SidebarConfig


def render_sidebar() -> SidebarConfig:
    st.sidebar.header("Graph and algorithm Configuration")

    source = st.sidebar.radio("Graph source", ["Generate", "Upload file"])

    if source == "Generate":
        _render_generate_graph_controls()
    else:
        _render_upload_graph_controls()

    st.sidebar.markdown("---")
    st.sidebar.header("Simulation Parameters")
    threshold = st.sidebar.number_input("Bootstrap threshold (k)", 1, 50, 2)
    seed_fraction = st.sidebar.slider(
        "Initial infection probability", 0.01, 1.0, 0.05, 0.01
    )
    num_trials = st.sidebar.number_input("Number of trials", 10, 500, 50, step=10)

    return SidebarConfig(
        threshold=int(threshold),
        seed_fraction=float(seed_fraction),
        num_trials=int(num_trials),
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

