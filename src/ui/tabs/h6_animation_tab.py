from __future__ import annotations

import networkx as nx
import streamlit as st

from simulation.H6 import H6Model
from simulation.seed_selection import select_seeds
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade


def render_h6_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("H6 Probabilistic Threshold Animation")

    if graph.number_of_nodes() > 300:
        st.warning(
            "Animation works best with smaller graphs (<= 300 nodes). "
            "The current graph has %d nodes — layout may be slow." % graph.number_of_nodes()
        )

    if st.button("▶ Animate cascade", key="h6_run_anim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))
        sim = H6Model(graph, phi=config.phi, gamma=config.gamma)
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

        with st.spinner("Running simulation & building animation…"):
            result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
            fig = animate_cascade(graph, activation_sequence, show=False)

        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"Cascade infected {result.cascade_size}/{n} nodes "
            f"({result.cascade_fraction:.2%}) in {result.time_to_cascade} round(s)."
        )
