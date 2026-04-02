from __future__ import annotations

import networkx as nx
import streamlit as st

from simulation.H1 import H1Model
from simulation.seed_selection import select_seeds
from ui.state import SessionKeys, SidebarConfig


def render_h1_simulation_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("H1 Hybrid Cascade Results")

    if st.button("▶ Run simulation", key="h1_run_sim"):
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))

        sim = H1Model(graph, threshold=config.threshold, beta=config.beta, gamma=config.gamma)

        seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))
        result, _ = sim.run(seed_nodes)

        with st.spinner("Computing averaged metrics…"):
            metrics = sim.collect_metrics(
                seed_size, num_trials=config.num_trials, seed=42, strategy=config.seed_strategy
            )

        st.session_state[SessionKeys.H1_SIM_RESULTS] = {
            "result": result,
            "metrics": metrics,
            "seed_size": seed_size,
            "n": n,
        }

    if SessionKeys.H1_SIM_RESULTS not in st.session_state:
        return

    sr = st.session_state[SessionKeys.H1_SIM_RESULTS]
    result = sr["result"]
    metrics = sr["metrics"]
    n = sr["n"]

    st.markdown("#### Single-run result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cascade Fraction", f"{result.cascade_fraction:.4f}")
    c2.metric("Rounds", result.time_to_cascade)
    c3.metric("Large Cascade?", "✅" if result.is_large_cascade else "❌")

    robustness = (1 - result.cascade_fraction) * (1 / (1 + result.time_to_cascade))
    st.metric("Robustness Score (0 = fragile → 1 = robust)", f"{robustness:.4f}")

    st.markdown("---")
    st.markdown(f"#### Averaged metrics ({config.num_trials} trials)")

    if metrics.cascade_size == 0.0:
        st.warning("No cascade spread beyond the seeds with these parameters.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Cascade Size (avg fraction)", f"{metrics.cascade_size:.4f}")
    m2.metric("Critical Seed Size", metrics.critical_seed_size)
    m3.metric("Cascade Probability", f"{metrics.cascade_probability:.4f}")

    m4, m5 = st.columns(2)
    m4.metric("Time to Stabilise (avg rounds)", f"{metrics.time_to_cascade:.2f}")
    m5.metric("Cascade Threshold (seed fraction)", f"{metrics.cascade_threshold:.4f}")
