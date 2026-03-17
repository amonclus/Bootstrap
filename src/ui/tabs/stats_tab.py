from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.graph_statistics import compute_graph_statistics, degree_distribution
from ui.charts import apply_layout_geometry, build_edge_trace, resolve_positions


def render_stats_tab(graph: nx.Graph) -> None:
    st.subheader("Graph Visualisation")

    pos, is_lattice_graph, has_pos_attr = resolve_positions(graph)
    node_list = list(graph.nodes())
    n_nodes = graph.number_of_nodes()

    edge_width = 0.5 if nx.density(graph) > 0.15 else 0.8
    edge_trace = build_edge_trace(graph, pos, edge_width=edge_width)

    if is_lattice_graph:
        node_marker = dict(
            size=10,
            color="steelblue",
            symbol="square",
            line=dict(width=1, color="darkgray"),
        )
        hover_text = [f"({n[0]},{n[1]})" for n in node_list]
    elif has_pos_attr:
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
        degrees = [graph.degree(n) for n in node_list]
        node_marker = dict(
            size=int(max(4, min(8, 300 / n_nodes))),
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
    apply_layout_geometry(layout_kw, node_list, pos, is_lattice_graph, has_pos_attr)

    fig_graph = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(**layout_kw))
    st.plotly_chart(fig_graph, use_container_width=not (is_lattice_graph or has_pos_attr))

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

    st.subheader("Degree Distribution")
    dd = degree_distribution(graph)
    df_dd = pd.DataFrame(sorted(dd.items()), columns=["Degree", "Count"])
    fig_dd = px.bar(df_dd, x="Degree", y="Count", title="Degree Distribution")
    st.plotly_chart(fig_dd, use_container_width=True)

