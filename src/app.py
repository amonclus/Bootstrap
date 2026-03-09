from __future__ import annotations

import random
import tempfile
import os

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from input.graph_generator import (
    generate_er_graph,
    generate_random_geometric_graph,
    generate_lattice_graph,
)
from input.graph_loader import (
    load_graph_from_dimacs,
    load_graph_from_edge_list,
    load_graph_from_gml,
)
from simulation.bootstrap import BootstrapPercolation
from analysis.graph_statistics import compute_graph_statistics, degree_distribution
from analysis.parameter_sweep import (
    sweep_er_probability,
    sweep_geometric_radius,
    sweep_lattice_size,
    sweep_seed_fraction,
)
from visualization.visualization import animate_cascade

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bootstrap Percolation – Network Risk Analysis",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Bootstrap Percolation – Network Risk Analysis")
st.caption("A showcase tool for analysing cascade risk in networks.")

# ── Sidebar: graph configuration ─────────────────────────────────────────────

st.sidebar.header("Graph Configuration")

source = st.sidebar.radio("Graph source", ["Generate", "Upload file"])

graph: nx.Graph | None = None

if source == "Generate":
    graph_type = st.sidebar.selectbox(
        "Graph type", ["Erdős–Rényi", "Random Geometric", "Lattice"]
    )

    if graph_type == "Erdős–Rényi":
        n = st.sidebar.number_input("Number of nodes", 10, 5000, 100)
        p = st.sidebar.slider("Edge probability (p)", 0.01, 1.0, 0.1, 0.01)
    elif graph_type == "Random Geometric":
        n = st.sidebar.number_input("Number of nodes", 10, 5000, 100)
        radius = st.sidebar.slider("Connection radius (r)", 0.01, 1.0, 0.2, 0.01)
    else:  # Lattice
        grid_size = st.sidebar.number_input("Grid side length", 3, 100, 10)

    if st.sidebar.button("Generate graph"):
        with st.spinner("Generating graph…"):
            if graph_type == "Erdős–Rényi":
                graph = generate_er_graph(n, p)
            elif graph_type == "Random Geometric":
                graph = generate_random_geometric_graph(n, radius)
            else:
                graph = generate_lattice_graph(grid_size)
                graph = nx.convert_node_labels_to_integers(graph)
        st.session_state["graph"] = graph
        st.session_state.pop("sim_results", None)  # clear stale results
        st.sidebar.success(
            f"Graph created – {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

else:  # Upload file
    fmt = st.sidebar.selectbox("File format", ["DIMACS", "Edge List", "GML"])
    uploaded = st.sidebar.file_uploader("Upload graph file", type=["dimacs", "txt", "gml", "edgelist"])
    if uploaded is not None:
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
            st.session_state["graph"] = graph
            st.session_state.pop("sim_results", None)
            st.sidebar.success(
                f"Graph loaded – {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
        finally:
            os.unlink(tmp_path)

# Simulation parameters (always visible)
st.sidebar.markdown("---")
st.sidebar.header("Simulation Parameters")
threshold = st.sidebar.number_input("Bootstrap threshold (k)", 1, 50, 2)
seed_fraction = st.sidebar.slider(
    "Initial infection probability", 0.01, 1.0, 0.05, 0.01
)
num_trials = st.sidebar.number_input("Number of trials", 10, 500, 50, step=10)

# ── Retrieve graph from session state ────────────────────────────────────────

if "graph" in st.session_state:
    graph = st.session_state["graph"]

if graph is None:
    st.info("👈 Configure and generate (or upload) a graph in the sidebar to get started.")
    st.stop()

# ── Main area tabs ───────────────────────────────────────────────────────────

tab_stats, tab_sim, tab_anim, tab_sweep = st.tabs(
    ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation", "📈 Parameter Sweep"]
)

# ── Tab 1: Graph Statistics ──────────────────────────────────────────────────

with tab_stats:
    st.subheader("Structural Statistics")
    stats = compute_graph_statistics(graph)

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", stats["nodes"])
    col1.metric("Edges", stats["edges"])
    col1.metric("Density", f"{stats['density']:.4f}")

    col2.metric("Avg Degree", f"{stats['average_degree']:.2f}")
    col2.metric("Min Degree", stats["min_degree"])
    col2.metric("Max Degree", stats["max_degree"])

    col3.metric("Avg Clustering", f"{stats['average_clustering']:.4f}")
    col3.metric("Components", stats["num_components"])
    col3.metric("Diameter", stats["diameter"])

    st.metric("Avg Path Length", f"{stats['average_path_length']:.2f}")

    # Degree distribution chart
    st.subheader("Degree Distribution")
    dd = degree_distribution(graph)
    df_dd = pd.DataFrame(sorted(dd.items()), columns=["Degree", "Count"])
    fig_dd = px.bar(df_dd, x="Degree", y="Count", title="Degree Distribution")
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Tab 2: Cascade Simulation ───────────────────────────────────────────────

with tab_sim:
    st.subheader("Cascade Simulation Results")

    if st.button("▶ Run simulation", key="run_sim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(seed_fraction * n))

        sim = BootstrapPercolation(graph, threshold)

        # Single run
        seed_nodes = set(random.sample(list(graph.nodes()), seed_size))
        result, _ = sim.run(seed_nodes)

        # Averaged metrics
        with st.spinner("Computing averaged metrics…"):
            metrics = sim.collect_metrics(num_trials=num_trials, seed=42)

        st.session_state["sim_results"] = {
            "result": result,
            "metrics": metrics,
            "seed_size": seed_size,
            "n": n,
        }

    if "sim_results" in st.session_state:
        sr = st.session_state["sim_results"]
        result = sr["result"]
        metrics = sr["metrics"]
        n = sr["n"]

        st.markdown("#### Single-run result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cascade Fraction", f"{result.cascade_fraction:.4f}")
        c2.metric("Rounds", result.time_to_cascade)
        c3.metric("Full Cascade?", "✅" if result.is_full_cascade else "❌")

        robustness = (1 - result.cascade_fraction) * (1 / (1 + result.time_to_cascade))
        st.metric("Robustness Score (0 = fragile → 1 = robust)", f"{robustness:.4f}")

        st.markdown("---")
        st.markdown(f"#### Averaged metrics ({num_trials} trials)")

        if metrics.critical_seed_size == n:
            st.warning("Network is too sparse for cascades with this threshold.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Cascade Size (avg fraction)", f"{metrics.cascade_size:.4f}")
            m2.metric("Critical Seed Size", metrics.critical_seed_size)
            m3.metric("Cascade Probability", f"{metrics.cascade_probability:.4f}")

            m4, m5 = st.columns(2)
            m4.metric("Time to Cascade (avg rounds)", f"{metrics.time_to_cascade:.2f}")
            m5.metric("Percolation Threshold", f"{metrics.percolation_threshold:.4f}")

# ── Tab 3: Cascade Animation ────────────────────────────────────────────────

with tab_anim:
    st.subheader("Cascade Animation")

    if graph.number_of_nodes() > 300:
        st.warning(
            "Animation works best with smaller graphs (≤ 300 nodes). "
            "The current graph has %d nodes — layout may be slow." % graph.number_of_nodes()
        )

    if st.button("▶ Animate cascade", key="run_anim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(seed_fraction * n))
        sim = BootstrapPercolation(graph, threshold)
        seed_nodes = set(random.sample(list(graph.nodes()), seed_size))

        with st.spinner("Running simulation & building animation…"):
            result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
            fig = animate_cascade(graph, activation_sequence, show=False)

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Cascade infected {result.cascade_size}/{n} nodes "
            f"({result.cascade_fraction:.2%}) in {result.time_to_cascade} round(s)."
        )

# ── Tab 4: Parameter Sweep ──────────────────────────────────────────────────

with tab_sweep:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Erdős–Rényi probability", "Geometric radius", "Lattice size"],
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01)
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05)
        frac_steps = st.number_input("Number of steps", 3, 30, 10)
        fracs = [frac_min + i * (frac_max - frac_min) / (frac_steps - 1) for i in range(int(frac_steps))]

        if st.button("▶ Run sweep", key="sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = sweep_seed_fraction(graph, fracs, threshold=threshold, num_trials=sweep_trials)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            fig = px.line(
                df,
                x="seed_fraction",
                y="cascade_probability",
                markers=True,
                title="Cascade Probability vs Seed Fraction",
                labels={"seed_fraction": "Seed Fraction", "cascade_probability": "Cascade Probability"},
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(
                df,
                x="seed_fraction",
                y="cascade_size",
                markers=True,
                title="Cascade Size vs Seed Fraction",
                labels={"seed_fraction": "Seed Fraction", "cascade_size": "Avg Cascade Fraction"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif sweep_type == "Erdős–Rényi probability":
        er_n = st.number_input("Number of nodes", 10, 2000, 100, key="er_n")
        er_k = st.number_input("Threshold", 1, 20, 2, key="er_k")
        probs = st.text_input("Probabilities (comma-separated)", "0.01,0.05,0.1,0.2,0.3")
        prob_list = [float(x.strip()) for x in probs.split(",")]

        if st.button("▶ Run sweep", key="sweep_er"):
            with st.spinner("Sweeping ER probabilities…"):
                data = sweep_er_probability(er_n, prob_list, threshold=er_k, num_trials=sweep_trials)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            fig = px.line(df, x="p", y="cascade_probability", markers=True, title="Cascade Probability vs p")
            st.plotly_chart(fig, use_container_width=True)

    elif sweep_type == "Geometric radius":
        geo_n = st.number_input("Number of nodes", 10, 2000, 100, key="geo_n")
        geo_k = st.number_input("Threshold", 1, 20, 2, key="geo_k")
        radii_str = st.text_input("Radii (comma-separated)", "0.05,0.1,0.15,0.2,0.25,0.3")
        radii_list = [float(x.strip()) for x in radii_str.split(",")]

        if st.button("▶ Run sweep", key="sweep_geo"):
            with st.spinner("Sweeping geometric radii…"):
                data = sweep_geometric_radius(geo_n, radii_list, threshold=geo_k, num_trials=sweep_trials)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            fig = px.line(df, x="radius", y="cascade_probability", markers=True, title="Cascade Probability vs Radius")
            st.plotly_chart(fig, use_container_width=True)

    elif sweep_type == "Lattice size":
        lat_k = st.number_input("Threshold", 1, 20, 2, key="lat_k")
        sizes_str = st.text_input("Grid sizes (comma-separated)", "5,10,15,20")
        sizes_list = [int(x.strip()) for x in sizes_str.split(",")]

        if st.button("▶ Run sweep", key="sweep_lat"):
            with st.spinner("Sweeping lattice sizes…"):
                data = sweep_lattice_size(sizes_list, threshold=lat_k, num_trials=sweep_trials)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            fig = px.line(df, x="grid_size", y="cascade_probability", markers=True, title="Cascade Probability vs Grid Size")
            st.plotly_chart(fig, use_container_width=True)

