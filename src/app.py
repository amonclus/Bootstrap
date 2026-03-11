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
    layout="centered",
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

tab_stats, tab_sim, tab_vuln, tab_anim, tab_sweep = st.tabs(
    ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎯 Node Vulnerability",
     "🎬 Cascade Animation", "📈 Parameter Sweep"]
)

# ── Tab 1: Graph Statistics ──────────────────────────────────────────────────

with tab_stats:
    # ── Graph visualisation ──────────────────────────────────────────────
    st.subheader("Graph Visualisation")

    from visualization.visualization import _is_lattice

    is_lattice_graph = _is_lattice(graph)

    # — Pick the best layout for each graph type —
    # Random Geometric graphs carry natural 2-D positions in node attrs
    has_pos_attr = all("pos" in graph.nodes[n] for n in graph.nodes())

    if is_lattice_graph:
        pos = {n: (int(n[1]), -int(n[0])) for n in graph.nodes()}
    elif has_pos_attr:
        # Use the spatial coordinates that nx.random_geometric_graph stores
        pos = {n: tuple(graph.nodes[n]["pos"]) for n in graph.nodes()}
    else:
        # ER / generic graphs – Kamada-Kawai gives cleaner results for
        # small-to-medium graphs; fall back to spring for large ones.
        if graph.number_of_nodes() <= 500:
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.5 / (graph.number_of_nodes() ** 0.5), iterations=80)

    node_list = list(graph.nodes())

    # — Adapt visual parameters to graph density —
    density = nx.density(graph)
    n_nodes = graph.number_of_nodes()

    # Edge opacity: fade edges in dense graphs to reduce clutter
    edge_opacity = max(0.08, min(1.0, 0.6 / (1 + 20 * density)))
    edge_width = 0.5 if density > 0.15 else 0.8

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color=f"rgba(180,180,180,{edge_opacity})"),
        hoverinfo="none",
        mode="lines",
    )

    # — Node styling per graph type —
    if is_lattice_graph:
        node_marker = dict(
            size=10,
            color="steelblue",
            symbol="square",
            line=dict(width=1, color="darkgray"),
        )
        hover_text = [f"({n[0]},{n[1]})" for n in node_list]
    elif has_pos_attr:
        # Random Geometric – colour by degree to show connectivity
        degrees = [graph.degree(n) for n in node_list]
        node_marker = dict(
            size=7,
            color=degrees,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Degree", thickness=12, len=0.5),
            line=dict(width=0.5, color="white"),
        )
        hover_text = [f"Node {n}  (deg {graph.degree(n)})" for n in node_list]
    else:
        # ER / generic – colour by degree
        degrees = [graph.degree(n) for n in node_list]
        node_marker = dict(
            size=max(4, min(8, 300 / n_nodes)),
            color=degrees,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Degree", thickness=12, len=0.5),
            line=dict(width=0.5, color="white"),
        )
        hover_text = [f"Node {n}  (deg {graph.degree(n)})" for n in node_list]

    node_trace = go.Scatter(
        x=[pos[n][0] for n in node_list],
        y=[pos[n][1] for n in node_list],
        mode="markers",
        text=hover_text,
        hoverinfo="text",
        marker=node_marker,
    )

    layout_kw = dict(
        title="Network Structure",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    if is_lattice_graph:
        all_x = [pos[n][0] for n in node_list]
        all_y = [pos[n][1] for n in node_list]
        pad = 0.8
        layout_kw["xaxis"].update(
            range=[min(all_x) - pad, max(all_x) + pad],
            scaleanchor="y",
            scaleratio=1,
        )
        layout_kw["yaxis"].update(
            range=[min(all_y) - pad, max(all_y) + pad],
        )
        layout_kw["width"] = 700
        layout_kw["height"] = 700
    elif has_pos_attr:
        # Random Geometric lives in [0,1]² – keep aspect ratio square
        layout_kw["xaxis"].update(scaleanchor="y", scaleratio=1)
        layout_kw["width"] = 700
        layout_kw["height"] = 700

    fig_graph = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(**layout_kw))
    st.plotly_chart(fig_graph, use_container_width=not (is_lattice_graph or has_pos_attr))

    # ── Structural statistics ────────────────────────────────────────────
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

# ── Tab 3: Node Vulnerability ───────────────────────────────────────────────

with tab_vuln:
    st.subheader("Node Vulnerability Analysis")
    st.markdown(
        "Two complementary views of node vulnerability:\n\n"
        "- **Activation** — Which nodes, if *infected*, trigger the largest "
        "cascades? (What to fear)\n"
        "- **Blocking** — Which nodes, if *immunised / removed*, reduce "
        "cascade spread the most? (What to protect)"
    )

    vuln_trials = st.number_input(
        "Trials per node", 5, 100, 15, step=5, key="vuln_trials",
        help="More trials → more stable scores, but slower.",
    )

    n_graph = graph.number_of_nodes()
    if n_graph > 500:
        st.warning(
            f"The graph has {n_graph} nodes — analysis requires "
            f"{n_graph}×{vuln_trials} = {n_graph * vuln_trials} simulations and "
            f"may take a while."
        )

    st.markdown("---")
    st.markdown("###Node Activation Analysis")
    st.caption("Each node is forced into the seed set across many trials to measure its influence on cascade size.")

    if st.button("▶ Run vulnerability analysis", key="run_vuln"):
        sim = BootstrapPercolation(graph, threshold)
        progress_bar = st.progress(0, text="Evaluating nodes…")

        def _update_progress(current, total):
            progress_bar.progress(current / total, text=f"Evaluating node {current}/{total}…")

        vuln_data = sim.node_influence_analysis(
            seed_fraction=seed_fraction,
            num_trials=vuln_trials,
            seed=42,
            progress_callback=_update_progress,
        )
        progress_bar.empty()

        st.session_state["vuln_data"] = vuln_data

    if "vuln_data" in st.session_state:
        vuln_data = st.session_state["vuln_data"]
        df_vuln = pd.DataFrame(vuln_data)

        # ── Graph coloured by influence score ────────────────────────────
        st.subheader("Influence Map")
        st.caption("🔴 Red = weak (high influence)  ·  🔵 Blue = strong (low influence)")

        from visualization.visualization import _is_lattice

        is_lattice_graph = _is_lattice(graph)
        has_pos_attr = all("pos" in graph.nodes[nd] for nd in graph.nodes())

        if is_lattice_graph:
            v_pos = {nd: (int(nd[1]), -int(nd[0])) for nd in graph.nodes()}
        elif has_pos_attr:
            v_pos = {nd: tuple(graph.nodes[nd]["pos"]) for nd in graph.nodes()}
        else:
            if graph.number_of_nodes() <= 500:
                v_pos = nx.kamada_kawai_layout(graph)
            else:
                v_pos = nx.spring_layout(
                    graph, seed=42,
                    k=1.5 / (graph.number_of_nodes() ** 0.5), iterations=80,
                )

        # Build node → metric lookups
        score_map = {row["node"]: row["influence_score"] for row in vuln_data}
        cp_map = {row["node"]: row["cascade_probability"] for row in vuln_data}
        time_map = {row["node"]: row["avg_time"] for row in vuln_data}
        v_node_list = list(graph.nodes())
        scores = [score_map[nd] for nd in v_node_list]

        # Edges
        v_edge_x, v_edge_y = [], []
        for u, v in graph.edges():
            v_edge_x += [v_pos[u][0], v_pos[v][0], None]
            v_edge_y += [v_pos[u][1], v_pos[v][1], None]

        density = nx.density(graph)
        edge_op = max(0.08, min(1.0, 0.6 / (1 + 20 * density)))

        v_edge_trace = go.Scatter(
            x=v_edge_x, y=v_edge_y,
            line=dict(width=0.5, color=f"rgba(180,180,180,{edge_op})"),
            hoverinfo="none", mode="lines",
        )

        hover_labels = [
            f"Node {nd}<br>"
            f"Influence: {score_map[nd]:.4f}<br>"
            f"Cascade Prob: {cp_map[nd]:.2%}<br>"
            f"Avg Time: {time_map[nd]:.1f} rounds<br>"
            f"Degree: {graph.degree(nd)}"
            for nd in v_node_list
        ]

        v_node_trace = go.Scatter(
            x=[v_pos[nd][0] for nd in v_node_list],
            y=[v_pos[nd][1] for nd in v_node_list],
            mode="markers",
            text=hover_labels,
            hoverinfo="text",
            marker=dict(
                size=10 if is_lattice_graph else max(5, min(10, 400 / n_graph)),
                color=scores,
                colorscale="RdBu_r",       # red = high (weak), blue = low (strong)
                showscale=True,
                colorbar=dict(title="Influence", thickness=12, len=0.6),
                symbol="square" if is_lattice_graph else "circle",
                line=dict(width=0.5, color="white"),
            ),
        )

        v_layout_kw = dict(
            title="Node Influence Map",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        if is_lattice_graph:
            all_vx = [v_pos[nd][0] for nd in v_node_list]
            all_vy = [v_pos[nd][1] for nd in v_node_list]
            pad = 0.8
            v_layout_kw["xaxis"].update(
                range=[min(all_vx) - pad, max(all_vx) + pad],
                scaleanchor="y", scaleratio=1,
            )
            v_layout_kw["yaxis"].update(
                range=[min(all_vy) - pad, max(all_vy) + pad],
            )
            v_layout_kw["width"] = 700
            v_layout_kw["height"] = 700
        elif has_pos_attr:
            v_layout_kw["xaxis"].update(scaleanchor="y", scaleratio=1)
            v_layout_kw["width"] = 700
            v_layout_kw["height"] = 700

        fig_vuln = go.Figure(
            data=[v_edge_trace, v_node_trace],
            layout=go.Layout(**v_layout_kw),
        )
        st.plotly_chart(fig_vuln, use_container_width=not (is_lattice_graph or has_pos_attr))

        # ── Weakest & strongest tables ───────────────────────────────────
        top_n = min(10, len(df_vuln))

        col_weak, col_strong = st.columns(2)

        with col_weak:
            st.markdown("#### 🔴 Top Weakest Nodes")
            st.caption("Highest influence — infecting these triggers the largest cascades.")
            st.dataframe(
                df_vuln.head(top_n).style.format({
                    "influence_score": "{:.4f}",
                    "cascade_probability": "{:.2%}",
                    "avg_time": "{:.1f}",
                    "cascade_std": "{:.4f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with col_strong:
            st.markdown("#### 🔵 Top Strongest Nodes")
            st.caption("Lowest influence — infecting these barely spreads.")
            st.dataframe(
                df_vuln.tail(top_n).iloc[::-1].style.format({
                    "influence_score": "{:.4f}",
                    "cascade_probability": "{:.2%}",
                    "avg_time": "{:.1f}",
                    "cascade_std": "{:.4f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Full ranked table (collapsible) ──────────────────────────────
        with st.expander("📋 Full node ranking"):
            st.dataframe(
                df_vuln.style.format({
                    "influence_score": "{:.4f}",
                    "cascade_probability": "{:.2%}",
                    "avg_time": "{:.1f}",
                    "cascade_std": "{:.4f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Influence vs. structural metrics scatter plots ───────────────
        st.subheader("Influence vs Structural Metrics")
        metric_choice = st.selectbox(
            "Compare influence against:",
            ["degree", "betweenness", "closeness", "cascade_probability", "avg_time", "cascade_std"],
            key="vuln_metric",
        )
        fig_corr = px.scatter(
            df_vuln,
            x=metric_choice,
            y="influence_score",
            hover_data=["node", "degree", "betweenness", "closeness"],
            title=f"Influence Score vs {metric_choice.title()}",
            labels={"influence_score": "Influence Score", metric_choice: metric_choice.title()},
            trendline="ols",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Node Blocking / Immunisation Analysis ────────────────────────────
    st.markdown("---")
    st.markdown("###Node Blocking Analysis")
    st.caption(
        "Each node is removed (immunised) from the graph and bootstrap "
        "percolation is re-run.  Nodes whose removal causes the biggest "
        "cascade reduction are the most critical to protect."
    )

    block_trials = st.number_input(
        "Trials per node (blocking)", 5, 100, 15, step=5, key="block_trials",
        help="More trials → more stable scores, but slower.",
    )

    if st.button("▶ Run blocking analysis", key="run_block"):
        sim = BootstrapPercolation(graph, threshold)
        progress_bar_b = st.progress(0, text="Evaluating node removals…")

        def _update_block_progress(current, total):
            progress_bar_b.progress(
                current / total,
                text=f"Blocking node {current}/{total}…",
            )

        block_data, bl_avg, bl_prob = sim.node_blocking_analysis(
            seed_fraction=seed_fraction,
            num_trials=block_trials,
            seed=42,
            progress_callback=_update_block_progress,
        )
        progress_bar_b.empty()

        st.session_state["block_data"] = block_data
        st.session_state["block_baseline"] = {"avg": bl_avg, "prob": bl_prob}

    if "block_data" in st.session_state:
        block_data = st.session_state["block_data"]
        bl_baseline = st.session_state["block_baseline"]
        df_block = pd.DataFrame(block_data)

        # Baseline summary
        st.info(
            f"**Baseline** (original graph): avg cascade fraction "
            f"**{bl_baseline['avg']:.4f}**, full-cascade probability "
            f"**{bl_baseline['prob']:.2%}**"
        )

        # ── Graph coloured by blocking effectiveness ─────────────────
        st.subheader("Blocking Effectiveness Map")
        st.caption("🟢 Green = critical to protect  ·  ⚪ Gray = low blocking impact")

        from visualization.visualization import _is_lattice

        is_lattice_graph_b = _is_lattice(graph)
        has_pos_attr_b = all("pos" in graph.nodes[nd] for nd in graph.nodes())

        if is_lattice_graph_b:
            b_pos = {nd: (int(nd[1]), -int(nd[0])) for nd in graph.nodes()}
        elif has_pos_attr_b:
            b_pos = {nd: tuple(graph.nodes[nd]["pos"]) for nd in graph.nodes()}
        else:
            if graph.number_of_nodes() <= 500:
                b_pos = nx.kamada_kawai_layout(graph)
            else:
                b_pos = nx.spring_layout(
                    graph, seed=42,
                    k=1.5 / (graph.number_of_nodes() ** 0.5), iterations=80,
                )

        red_map = {row["node"]: row["cascade_reduction"] for row in block_data}
        prob_red_map = {row["node"]: row["prob_reduction"] for row in block_data}
        b_node_list = list(graph.nodes())
        reductions = [red_map[nd] for nd in b_node_list]

        # Edges
        b_edge_x, b_edge_y = [], []
        for u, v in graph.edges():
            b_edge_x += [b_pos[u][0], b_pos[v][0], None]
            b_edge_y += [b_pos[u][1], b_pos[v][1], None]

        b_density = nx.density(graph)
        b_edge_op = max(0.08, min(1.0, 0.6 / (1 + 20 * b_density)))

        b_edge_trace = go.Scatter(
            x=b_edge_x, y=b_edge_y,
            line=dict(width=0.5, color=f"rgba(180,180,180,{b_edge_op})"),
            hoverinfo="none", mode="lines",
        )

        b_hover = [
            f"Node {nd}<br>"
            f"Cascade Reduction: {red_map[nd]:.4f}<br>"
            f"Prob Reduction: {prob_red_map[nd]:.2%}<br>"
            f"Degree: {graph.degree(nd)}"
            for nd in b_node_list
        ]

        n_graph_b = graph.number_of_nodes()
        b_node_trace = go.Scatter(
            x=[b_pos[nd][0] for nd in b_node_list],
            y=[b_pos[nd][1] for nd in b_node_list],
            mode="markers",
            text=b_hover,
            hoverinfo="text",
            marker=dict(
                size=10 if is_lattice_graph_b else max(5, min(10, 400 / n_graph_b)),
                color=reductions,
                colorscale="Greens",
                showscale=True,
                colorbar=dict(title="Cascade<br>Reduction", thickness=12, len=0.6),
                symbol="square" if is_lattice_graph_b else "circle",
                line=dict(width=0.5, color="white"),
            ),
        )

        b_layout_kw = dict(
            title="Blocking Effectiveness Map",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        if is_lattice_graph_b:
            all_bx = [b_pos[nd][0] for nd in b_node_list]
            all_by = [b_pos[nd][1] for nd in b_node_list]
            pad = 0.8
            b_layout_kw["xaxis"].update(
                range=[min(all_bx) - pad, max(all_bx) + pad],
                scaleanchor="y", scaleratio=1,
            )
            b_layout_kw["yaxis"].update(
                range=[min(all_by) - pad, max(all_by) + pad],
            )
            b_layout_kw["width"] = 700
            b_layout_kw["height"] = 700
        elif has_pos_attr_b:
            b_layout_kw["xaxis"].update(scaleanchor="y", scaleratio=1)
            b_layout_kw["width"] = 700
            b_layout_kw["height"] = 700

        fig_block = go.Figure(
            data=[b_edge_trace, b_node_trace],
            layout=go.Layout(**b_layout_kw),
        )
        st.plotly_chart(fig_block, use_container_width=not (is_lattice_graph_b or has_pos_attr_b))

        # ── Most / least critical tables ─────────────────────────────
        top_b = min(10, len(df_block))

        col_crit, col_low = st.columns(2)

        with col_crit:
            st.markdown("#### 🟢 Most Critical to Protect")
            st.caption("Blocking these nodes reduces cascades the most.")
            st.dataframe(
                df_block.head(top_b).style.format({
                    "cascade_reduction": "{:.4f}",
                    "prob_reduction": "{:.2%}",
                    "cascade_blocked": "{:.4f}",
                    "prob_blocked": "{:.2%}",
                    "time_blocked": "{:.1f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with col_low:
            st.markdown("#### ⚪ Least Critical to Protect")
            st.caption("Blocking these nodes barely changes cascade behaviour.")
            st.dataframe(
                df_block.tail(top_b).iloc[::-1].style.format({
                    "cascade_reduction": "{:.4f}",
                    "prob_reduction": "{:.2%}",
                    "cascade_blocked": "{:.4f}",
                    "prob_blocked": "{:.2%}",
                    "time_blocked": "{:.1f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("📋 Full blocking ranking"):
            st.dataframe(
                df_block.style.format({
                    "cascade_reduction": "{:.4f}",
                    "prob_reduction": "{:.2%}",
                    "cascade_blocked": "{:.4f}",
                    "prob_blocked": "{:.2%}",
                    "time_blocked": "{:.1f}",
                    "betweenness": "{:.4f}",
                    "closeness": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Blocking effectiveness vs structural metrics ─────────────
        st.subheader("Blocking Effectiveness vs Structural Metrics")
        block_metric = st.selectbox(
            "Compare cascade reduction against:",
            ["degree", "betweenness", "closeness", "prob_reduction", "time_blocked"],
            key="block_metric",
        )
        fig_block_corr = px.scatter(
            df_block,
            x=block_metric,
            y="cascade_reduction",
            hover_data=["node", "degree", "betweenness", "closeness"],
            title=f"Cascade Reduction vs {block_metric.replace('_', ' ').title()}",
            labels={
                "cascade_reduction": "Cascade Reduction",
                block_metric: block_metric.replace("_", " ").title(),
            },
            trendline="ols",
        )
        st.plotly_chart(fig_block_corr, use_container_width=True)

# ── Tab 4: Cascade Animation ────────────────────────────────────────────────

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

# ── Tab 5: Parameter Sweep ──────────────────────────────────────────────────

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

