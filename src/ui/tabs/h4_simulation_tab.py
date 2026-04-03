from __future__ import annotations

import networkx as nx
import plotly.express as px
import streamlit as st

from simulation.H4 import H4Model
from simulation.seed_selection import select_seeds
from ui.state import SessionKeys, SidebarConfig


def render_h4_simulation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("H4 Hybrid Cascade Results")

    if st.button("▶ Run simulation", key="h4_run_sim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))

        sim = H4Model(graph, phi=config.phi, beta=config.beta, gamma=config.gamma)
        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))
        result, _ = sim.run(seed_nodes)

        with st.spinner("Computing averaged metrics…"):
            metrics = sim.collect_metrics(
                seed_size, num_trials=config.num_trials, seed=42, strategy=config.seed_strategy
            )

        st.session_state[SessionKeys.H4_SIM_RESULTS] = {
            "result": result, "metrics": metrics, "seed_size": seed_size, "n": n,
        }

    if SessionKeys.H4_SIM_RESULTS not in st.session_state:
        return

    sr = st.session_state[SessionKeys.H4_SIM_RESULTS]
    result = sr["result"]
    metrics = sr["metrics"]
    n = sr["n"]

    st.markdown("#### Single-run result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Infected Fraction", f"{result.cascade_fraction:.4f}")
    c2.metric("Rounds", result.time_to_cascade)
    c3.metric("Large Cascade?", "✅" if result.is_large_cascade else "❌")

    if result.infected_series:
        fig = px.line(
            x=list(range(len(result.infected_series))),
            y=result.infected_series,
            labels={"x": "Round", "y": "Infected nodes"},
            title="H4 Epidemic Curve (single run)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### Averaged metrics ({config.num_trials} trials)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Cascade Size (avg peak fraction)", f"{metrics.cascade_size:.4f}")
    m2.metric("Critical Seed Size", metrics.critical_seed_size)
    m3.metric("Cascade Probability", f"{metrics.cascade_probability:.4f}")

    m4, m5 = st.columns(2)
    m4.metric("Time to Stabilise (avg rounds)", f"{metrics.time_to_cascade:.2f}")
    m5.metric("Cascade Threshold (seed fraction)", f"{metrics.cascade_threshold:.4f}")
