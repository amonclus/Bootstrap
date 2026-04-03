from __future__ import annotations

import networkx as nx
import streamlit as st

from simulation.H5 import H5Model
from simulation.seed_selection import select_seeds
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade


def render_h5_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("H5 Sequential Hybrid Animation")

    if graph.number_of_nodes() > 300:
        st.warning(
            "Animation works best with smaller graphs (<= 300 nodes). "
            "The current graph has %d nodes — layout may be slow." % graph.number_of_nodes()
        )

    if st.button("▶ Animate cascade", key="h5_run_anim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))
        sim = H5Model(graph, phi=config.phi, beta=config.beta, gamma=config.gamma,
                      switch_fraction=config.switch_fraction)
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

        with st.spinner("Running simulation & building animation…"):
            result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
            fig = animate_cascade(graph, activation_sequence, show=False)

        st.plotly_chart(fig, use_container_width=True)
        phase_info = (
            f"Phase 1 (SIS): {result.rounds_phase1} rounds → "
            f"Phase 2 (WTM): {result.rounds_phase2} rounds."
            if result.switched else
            f"Switch threshold not reached — ran as pure SIS ({result.rounds_phase1} rounds)."
        )
        st.info(
            f"Cascade infected {result.cascade_size}/{n} nodes "
            f"({result.cascade_fraction:.2%}). {phase_info}"
        )
