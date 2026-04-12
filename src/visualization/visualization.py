"""
Visualization module
Responsible for making diagrams and animations of the bootstrap process.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx

import plotly.graph_objects as go


def _is_lattice(graph: nx.Graph) -> bool:
    """Detect whether the graph is a 2-D lattice (nodes are (row, col) tuples).
    Args:
        graph: nx.Graph
    Returns:
        bool
    """
    if graph.number_of_nodes() == 0:
        return False
    sample_node = next(iter(graph.nodes()))
    return isinstance(sample_node, tuple) and len(sample_node) == 2


def animate_cascade(graph: nx.Graph, activation_sequence: List[set], save_path: Optional[str] = None, show: bool = True):
    """
        Creates an animation of the cascade process using Plotly. Each frame corresponds to one round of the cascade, showing which nodes are activated.
    Args:
        graph: nx.Graph
        activation_sequence: List of sets, where each set contains the nodes activated in that round of the cascade. The first set should be the initial seeds.
        save_path: Path to save the animation HTML file (optional)
        show: If True the animation will be shown on the screen, otherwise it will only be saved (optional)
    """
    is_lattice = _is_lattice(graph)
    has_pos_attr = all("pos" in graph.nodes[n] for n in graph.nodes())

    # Choose layout
    if is_lattice:
        pos = {n: (int(n[1]), -int(n[0])) for n in graph.nodes()}
    elif has_pos_attr:
        pos = {n: tuple(graph.nodes[n]["pos"]) for n in graph.nodes()}
    else:
        if graph.number_of_nodes() <= 500:
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.5 / (graph.number_of_nodes() ** 0.5), iterations=80)
    node_list = list(graph.nodes())

    # Adjust marker & label settings
    if is_lattice:
        marker_size = 12
        node_symbol = 'square'
        marker_line = dict(width=1, color='darkgray')
        node_text = [f"({n[0]},{n[1]})" for n in node_list]
    else:
        marker_size = max(4, min(8, 300 / graph.number_of_nodes()))
        node_symbol = 'circle'
        marker_line = dict(width=0.5, color='white')
        node_text = [str(n) for n in node_list]

    # Adaptive edge styling
    density = nx.density(graph)
    edge_opacity = max(0.08, min(1.0, 0.6 / (1 + 20 * density)))
    edge_width = 0.5 if density > 0.15 else 0.8

    # Create base node trace
    node_trace = go.Scatter(
        x=[pos[n][0] for n in node_list],
        y=[pos[n][1] for n in node_list],
        mode='markers',
        text=node_text,
        marker=dict(
            size=marker_size,
            color='lightgray',
            symbol=node_symbol,
            line=marker_line,
        ),
        hoverinfo='text',
    )

    # Create edge traces
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color=f'rgba(180,180,180,{edge_opacity})'),
        hoverinfo='none',
        mode='lines'
    )

    # Build frames – each frame must update the node trace (index 1)
    # activation_sequence entries are either set (bootstrap) or (newly_infected, newly_recovered) tuple (SIR)
    is_sir = activation_sequence and isinstance(activation_sequence[0], tuple)

    frames = []
    ever_infected: set = set()
    recovered_so_far: set = set()
    for i, entry in enumerate(activation_sequence):
        if is_sir:
            newly_infected, newly_recovered = entry
            ever_infected |= newly_infected
            recovered_so_far |= newly_recovered
            def _color(n):
                if n in recovered_so_far:
                    return 'green'
                if n in ever_infected:
                    return 'red'
                return 'lightgray'
            node_colors = [_color(n) for n in node_list]
        else:
            ever_infected |= entry
            node_colors = ['red' if n in ever_infected else 'lightgray' for n in node_list]

        frames.append(go.Frame(
            data=[
                edge_trace,  # keep edges unchanged
                go.Scatter(
                    x=[pos[n][0] for n in node_list],
                    y=[pos[n][1] for n in node_list],
                    mode='markers',
                    text=node_text,
                    marker=dict(
                        size=marker_size,
                        color=node_colors,
                        symbol=node_symbol,
                        line=marker_line,
                    ),
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

    # Build layout – enforce equal aspect ratio for spatial graphs
    layout_kwargs = dict(
        title='Cascade Animation',
        showlegend=False,
        sliders=sliders,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(
            type='buttons',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True, transition=dict(duration=300))]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]),
            ],
        )],
    )

    if is_lattice:
        all_x = [pos[n][0] for n in node_list]
        all_y = [pos[n][1] for n in node_list]
        pad = 0.8
        layout_kwargs["xaxis"].update(
            range=[min(all_x) - pad, max(all_x) + pad],
            scaleanchor='y', scaleratio=1,
        )
        layout_kwargs["yaxis"].update(
            range=[min(all_y) - pad, max(all_y) + pad],
        )
        layout_kwargs["width"] = 700
        layout_kwargs["height"] = 700
    elif has_pos_attr:
        layout_kwargs["xaxis"].update(scaleanchor='y', scaleratio=1)
        layout_kwargs["width"] = 700
        layout_kwargs["height"] = 700

    # Build figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(**layout_kwargs),
        frames=frames,
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()

    return fig


def plot_phase_transition(sweep_results: List[Dict], x_param: str, y_param: str, xlabel: str, ylabel: str, title: str, threshold_param: Optional[str] = None, figsize=(8, 5),save_path: Optional[str] = None,):
    """
    Plots the phase transition
    Args:
        sweep_results: Results of the parameter sweep
        x_param: Parameter (x-axis)
        y_param: Parameter (y-axis)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Title of chart
        threshold_param: Parameter of the threshold (optional, if provided will plot separate lines for each threshold value)
        figsize: Size of the figure
        save_path: Path to save the figure
    """
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


def plot_cascade_evolution( cascade_fractions: List[float], rounds: List[int], xlabel="Rounds", ylabel="Cascade fraction", title="Cascade evolution", figsize=(8, 5), save_path: Optional[str] = None,):
    """
    Plots the cascade evolution
    Args:
        cascade_fractions: Fractions for each round
        rounds: Rounds of cascade
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Title of chart
        figsize: Size of the figure
        save_path: Path to save the figure

    """
    plt.figure(figsize=figsize)
    plt.plot(rounds, cascade_fractions, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()