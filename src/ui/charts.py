from __future__ import annotations

from typing import Any

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from visualization.visualization import _is_lattice


def _hash_graph(g: nx.Graph) -> int:
    """Structural hash for a networkx Graph usable with st.cache_data."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    edges = list(g.edges())
    # Sample at most 2000 edges for large graphs to keep hashing fast
    step = max(1, m // 2000)
    return hash((n, m, frozenset(edges[::step])))


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_graph})
def resolve_positions(graph: nx.Graph) -> tuple[dict[Any, tuple[float, float]], bool, bool]:
    is_lattice_graph = _is_lattice(graph)
    has_pos_attr = all("pos" in graph.nodes[n] for n in graph.nodes())

    if is_lattice_graph:
        pos = {n: (int(n[1]), -int(n[0])) for n in graph.nodes()}
    elif has_pos_attr:
        pos = {n: tuple(graph.nodes[n]["pos"]) for n in graph.nodes()}
    elif graph.number_of_nodes() <= 500:
        pos = nx.kamada_kawai_layout(graph)
    else:
        n = graph.number_of_nodes()
        iterations = 30 if n > 2000 else 50 if n > 1000 else 80
        pos = nx.spring_layout(
            graph,
            seed=42,
            k=1.5 / (n ** 0.5),
            iterations=iterations,
        )

    return pos, is_lattice_graph, has_pos_attr


def build_edge_trace(graph: nx.Graph, pos: dict[Any, tuple[float, float]], edge_width: float = 0.5) -> go.Scatter:
    density = nx.density(graph)
    edge_opacity = max(0.08, min(1.0, 0.6 / (1 + 20 * density)))

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=edge_width, color=f"rgba(180,180,180,{edge_opacity})"),
        hoverinfo="none",
        mode="lines",
    )


def apply_layout_geometry(
    layout_kw: dict[str, Any],
    node_list: list[Any],
    pos: dict[Any, tuple[float, float]],
    is_lattice_graph: bool,
    has_pos_attr: bool,
) -> None:
    if is_lattice_graph:
        all_x = [pos[n][0] for n in node_list]
        all_y = [pos[n][1] for n in node_list]
        pad = 0.8
        layout_kw["xaxis"].update(
            range=[min(all_x) - pad, max(all_x) + pad],
            scaleanchor="y",
            scaleratio=1,
        )
        layout_kw["yaxis"].update(range=[min(all_y) - pad, max(all_y) + pad])
        layout_kw["width"] = 700
        layout_kw["height"] = 700
    elif has_pos_attr:
        layout_kw["xaxis"].update(scaleanchor="y", scaleratio=1)
        layout_kw["width"] = 700
        layout_kw["height"] = 700

