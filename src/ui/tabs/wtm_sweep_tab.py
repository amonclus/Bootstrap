from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.wtm_parameter_sweep import wtm_sweep_phi, wtm_sweep_seed_fraction
from ui.state import SidebarConfig


def render_wtm_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Fractional threshold (φ)"],
        key="wtm_sweep_type",
    )
    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="wtm_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="wtm_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="wtm_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="wtm_frac_steps")
        fracs = [frac_min + i * (frac_max - frac_min) / (frac_steps - 1) for i in range(int(frac_steps))]

        if st.button("▶ Run sweep", key="wtm_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = wtm_sweep_seed_fraction(
                    graph, fracs, phi=config.phi,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="seed_fraction", xlabel="Seed Fraction")

    elif sweep_type == "Fractional threshold (φ)":
        phi_min = st.slider("Min φ", 0.01, 0.5, 0.05, 0.01, key="wtm_phi_min")
        phi_max = st.slider("Max φ", 0.1, 1.0, 0.9, 0.05, key="wtm_phi_max")
        phi_steps = st.number_input("Number of steps", 3, 30, 10, key="wtm_phi_steps")
        phis = [phi_min + i * (phi_max - phi_min) / (phi_steps - 1) for i in range(int(phi_steps))]

        if st.button("▶ Run sweep", key="wtm_sweep_phi_run"):
            with st.spinner("Sweeping fractional thresholds…"):
                data = wtm_sweep_phi(
                    graph, phis, seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="phi", xlabel="Fractional Threshold (φ)")


def _render_charts(df: pd.DataFrame, x: str, xlabel: str) -> None:
    st.dataframe(df, use_container_width=True)

    fig = px.line(df, x=x, y="cascade_probability", markers=True,
                  title=f"Cascade Probability vs {xlabel}",
                  labels={x: xlabel, "cascade_probability": "Cascade Probability"})
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(df, x=x, y="cascade_size", markers=True,
                   title=f"Cascade Size vs {xlabel}",
                   labels={x: xlabel, "cascade_size": "Avg Cascade Fraction"})
    st.plotly_chart(fig2, use_container_width=True)
