
from __future__ import annotations
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx

import plotly.graph_objects as go


def animate_cascade(graph: nx.Graph, activation_sequence: List[set], save_path: Optional[str] = None):

    pos = nx.spring_layout(graph, seed=42)
    node_list = list(graph.nodes())

    # Create base node trace
    node_trace = go.Scatter(
        x=[pos[n][0] for n in node_list],
        y=[pos[n][1] for n in node_list],
        mode='markers+text',
        text=[str(n) for n in node_list],
        textposition='top center',
        marker=dict(size=20, color='lightgray'),
        hoverinfo='text'
    )

    # Create edge traces
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Build frames – each frame must update the node trace (index 1)
    frames = []
    activated_so_far = set()
    for i, activated in enumerate(activation_sequence):
        activated_so_far |= activated
        node_colors = ['red' if n in activated_so_far else 'lightgray' for n in node_list]
        frames.append(go.Frame(
            data=[
                edge_trace,  # keep edges unchanged
                go.Scatter(
                    x=[pos[n][0] for n in node_list],
                    y=[pos[n][1] for n in node_list],
                    mode='markers+text',
                    text=[str(n) for n in node_list],
                    textposition='top center',
                    marker=dict(size=20, color=node_colors),
                    hoverinfo='text',
                ),
            ],
            name=str(i),
        ))

    # Build sliders for frame navigation
    sliders = [dict(
        active=0,
        steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                     method='animate', label=str(i))
               for i in range(len(frames))],
        currentvalue=dict(prefix='Round: '),
        pad=dict(t=50),
    )]

    # Build figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Cascade Animation',
            showlegend=False,
            sliders=sliders,
            updatemenus=[dict(
                type='buttons',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True, transition=dict(duration=300))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]),
                ],
            )],
        ),
        frames=frames,
    )

    if save_path:
        fig.write_html(save_path)
    fig.show()


def plot_phase_transition(
    sweep_results: List[Dict],
    x_param: str,
    y_param: str,
    xlabel: str,
    ylabel: str,
    title: str,
    threshold_param: Optional[str] = None,
    figsize=(8, 5),
    save_path: Optional[str] = None,
):

    plt.figure(figsize=figsize)

    if threshold_param:
        thresholds = sorted(set(entry[threshold_param] for entry in sweep_results))
        for k in thresholds:
            xs = [entry[x_param] for entry in sweep_results if entry[threshold_param] == k]
            ys = [entry[y_param] for entry in sweep_results if entry[threshold_param] == k]
            plt.plot(xs, ys, marker='o', label=f"{threshold_param}={k}")
    else:
        xs = [entry[x_param] for entry in sweep_results]
        ys = [entry[y_param] for entry in sweep_results]
        plt.plot(xs, ys, marker='o', label=y_param)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_cascade_evolution(
    cascade_fractions: List[float],
    rounds: List[int],
    xlabel="Rounds",
    ylabel="Cascade fraction",
    title="Cascade evolution",
    figsize=(8, 5),
    save_path: Optional[str] = None,
):

    plt.figure(figsize=figsize)
    plt.plot(rounds, cascade_fractions, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()