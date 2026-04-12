from __future__ import annotations

import networkx as nx
import streamlit as st

from simulation.seed_selection import select_seeds
from simulation.sir import SIRModel
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade

LARGE_GRAPH_THRESHOLD = 500


def render_sir_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Epidemic Animation")
    n = graph.number_of_nodes()
    large = n > LARGE_GRAPH_THRESHOLD

    if large:
        st.info(f"Graph has {n} nodes — animation is disabled. Simulation results will be shown as text.")

    if st.button("▶ Run simulation" if large else "▶ Animate epidemic", key="sir_run_anim"):
        seed_size = max(1, int(config.seed_fraction * n))
        sim = SIRModel(graph, beta=config.beta, gamma=config.gamma)
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

        if large:
            with st.spinner("Running simulation…"):
                result, _ = sim.run(seed_nodes, record_sequence=False)
        else:
            with st.spinner("Running simulation & building animation…"):
                result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
                fig = animate_cascade(graph, activation_sequence, show=False)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Epidemic reached {result.epidemic_size}/{n} nodes "
            f"({result.epidemic_fraction:.2%}) in {result.time_to_epidemic} round(s)."
        )
