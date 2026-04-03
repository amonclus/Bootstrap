from __future__ import annotations

import networkx as nx
import plotly.express as px
import streamlit as st

from simulation.sis import SISModel
from simulation.seed_selection import select_seeds
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade


def render_sis_animation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("SIS Epidemic Animation")
    st.caption(
        "The animation uses the (infected, recovered→susceptible) frame format. "
        "Because SIS allows re-infection, recovered nodes appear green but may "
        "be re-infected in later rounds (not shown distinctly in the network view). "
        "Use the epidemic curve below for a precise time-series view."
    )

    if graph.number_of_nodes() > 300:
        st.warning(
            "Animation works best with smaller graphs (<= 300 nodes). "
            "The current graph has %d nodes — layout may be slow." % graph.number_of_nodes()
        )

    if st.button("▶ Animate epidemic", key="sis_run_anim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))
        sim = SISModel(graph, beta=config.beta, gamma=config.gamma)
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

        with st.spinner("Running simulation & building animation…"):
            result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
            fig_anim = animate_cascade(graph, activation_sequence, show=False)
            fig_curve = px.line(
                x=list(range(len(result.infected_series))),
                y=result.infected_series,
                labels={"x": "Round", "y": "Infected nodes"},
                title="SIS Epidemic Curve",
            )

        st.plotly_chart(fig_anim, use_container_width=True)
        st.plotly_chart(fig_curve, use_container_width=True)

        st.info(
            f"Peak infected: {result.peak_infected}/{n} nodes "
            f"({result.cascade_fraction:.2%}) over {result.time_to_cascade} round(s)."
        )
