from __future__ import annotations

import networkx as nx
import streamlit as st

from simulation.H2 import H2Model
from simulation.seed_selection import select_seeds
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade

LARGE_GRAPH_THRESHOLD = 500


def render_h2_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("H2 Sequential Hybrid — Cascade Animation")
    st.caption(
        "Red nodes are infected, green nodes have recovered (Phase 1 only). "
        "The animation shows both the SIR phase and the bootstrap phase as a single sequence."
    )
    n = graph.number_of_nodes()
    large = n > LARGE_GRAPH_THRESHOLD

    if large:
        st.info(f"Graph has {n} nodes — animation is disabled. Simulation results will be shown as text.")

    if st.button("▶ Run simulation" if large else "▶ Animate cascade", key="h2_run_anim"):
        seed_size = max(1, int(config.seed_fraction * n))
        sim = H2Model(
            graph,
            threshold=config.threshold,
            beta=config.beta,
            gamma=config.gamma,
            switch_fraction=config.switch_fraction,
        )
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

        if large:
            with st.spinner("Running simulation…"):
                result, _ = sim.run(seed_nodes, record_sequence=False)
        else:
            with st.spinner("Running simulation & building animation…"):
                result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
                fig = animate_cascade(graph, activation_sequence, show=False)
            st.plotly_chart(fig, use_container_width=True)

        if result.switched:
            st.info(
                f"Switch triggered at {result.switch_fraction:.2%} infected "
                f"(after {result.rounds_phase1} SIR round(s)). "
                f"Bootstrap phase ran for {result.rounds_phase2} further round(s). "
                f"Final cascade: {result.cascade_size}/{n} nodes ({result.cascade_fraction:.2%})."
            )
        else:
            st.info(
                f"Switch threshold ({config.switch_fraction:.0%}) never reached — ran as pure SIR. "
                f"Final cascade: {result.cascade_size}/{n} nodes ({result.cascade_fraction:.2%}) "
                f"in {result.rounds_phase1} round(s)."
            )
