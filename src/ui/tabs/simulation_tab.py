from __future__ import annotations

import random

import networkx as nx
import streamlit as st

from simulation.bootstrap import BootstrapPercolation
from ui.state import SessionKeys, SidebarConfig


def render_simulation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Cascade Simulation Results")

    if st.button("▶ Run simulation", key="run_sim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))

        sim = BootstrapPercolation(graph, config.threshold)

        seed_nodes = set(random.sample(list(graph.nodes()), seed_size))
        result, _ = sim.run(seed_nodes)

        with st.spinner("Computing averaged metrics…"):
            metrics = sim.collect_metrics(seed_size, num_trials=config.num_trials, seed=42)

        st.session_state[SessionKeys.SIM_RESULTS] = {
            "result": result,
            "metrics": metrics,
            "seed_size": seed_size,
            "n": n,
        }

    if SessionKeys.SIM_RESULTS not in st.session_state:
        return

    sr = st.session_state[SessionKeys.SIM_RESULTS]
    result = sr["result"]
    metrics = sr["metrics"]
    n = sr["n"]

    st.markdown("#### Single-run result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cascade Fraction", f"{result.cascade_fraction:.4f}")
    c2.metric("Rounds", result.time_to_cascade)
    c3.metric("Full Cascade?", "✅" if result.is_full_cascade else "❌")

    robustness = (1 - result.cascade_fraction) * (1 / (1 + result.time_to_cascade))
    st.metric("Robustness Score (0 = fragile → 1 = robust)", f"{robustness:.4f}")

    st.markdown("---")
    st.markdown(f"#### Averaged metrics ({config.num_trials} trials)")

    if metrics.critical_seed_size == n:
        st.warning("Network is too sparse for cascades with this threshold.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Cascade Size (avg fraction)", f"{metrics.cascade_size:.4f}")
    m2.metric("Critical Seed Size", metrics.critical_seed_size)
    m3.metric("Cascade Probability", f"{metrics.cascade_probability:.4f}")

    m4, m5 = st.columns(2)
    m4.metric("Time to Cascade (avg rounds)", f"{metrics.time_to_cascade:.2f}")
    m5.metric("Percolation Threshold", f"{metrics.percolation_threshold:.4f}")

