from __future__ import annotations

import random

import networkx as nx
import streamlit as st

from simulation.bootstrap import BootstrapPercolation
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade


def render_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Cascade Animation")

    if graph.number_of_nodes() > 300:
        st.warning(
            "Animation works best with smaller graphs (<= 300 nodes). "
            "The current graph has %d nodes - layout may be slow." % graph.number_of_nodes()
        )

    if st.button("▶ Animate cascade", key="run_anim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))
        sim = BootstrapPercolation(graph, config.threshold)
        seed_nodes = set(random.sample(list(graph.nodes()), seed_size))

        with st.spinner("Running simulation & building animation…"):
            result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
            fig = animate_cascade(graph, activation_sequence, show=False)

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Cascade infected {result.cascade_size}/{n} nodes "
            f"({result.cascade_fraction:.2%}) in {result.time_to_cascade} round(s)."
        )

