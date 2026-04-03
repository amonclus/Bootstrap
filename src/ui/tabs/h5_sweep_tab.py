from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.h5_parameter_sweep import h5_sweep_beta, h5_sweep_phi, h5_sweep_seed_fraction
from ui.state import SidebarConfig


def render_h5_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Transmission rate (β)", "WTM threshold (φ)"],
        key="h5_sweep_type",
    )
    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="h5_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="h5_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="h5_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="h5_frac_steps")
        fracs = [frac_min + i * (frac_max - frac_min) / (frac_steps - 1) for i in range(int(frac_steps))]

        if st.button("▶ Run sweep", key="h5_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = h5_sweep_seed_fraction(
                    graph, fracs, phi=config.phi, beta=config.beta, gamma=config.gamma,
                    switch_fraction=config.switch_fraction,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="seed_fraction", xlabel="Seed Fraction")

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="h5_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.8, 0.05, key="h5_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 10, key="h5_beta_steps")
        betas = [beta_min + i * (beta_max - beta_min) / (beta_steps - 1) for i in range(int(beta_steps))]

        if st.button("▶ Run sweep", key="h5_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = h5_sweep_beta(
                    graph, betas, phi=config.phi, gamma=config.gamma,
                    switch_fraction=config.switch_fraction,
                    seed_fraction=config.seed_fraction, num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="beta", xlabel="Transmission Rate (β)")

    elif sweep_type == "WTM threshold (φ)":
        phi_min = st.slider("Min φ", 0.01, 0.5, 0.05, 0.01, key="h5_phi_min")
        phi_max = st.slider("Max φ", 0.1, 1.0, 0.9, 0.05, key="h5_phi_max")
        phi_steps = st.number_input("Number of steps", 3, 30, 10, key="h5_phi_steps")
        phis = [phi_min + i * (phi_max - phi_min) / (phi_steps - 1) for i in range(int(phi_steps))]

        if st.button("▶ Run sweep", key="h5_sweep_phi_run"):
            with st.spinner("Sweeping WTM thresholds…"):
                data = h5_sweep_phi(
                    graph, phis, beta=config.beta, gamma=config.gamma,
                    switch_fraction=config.switch_fraction,
                    seed_fraction=config.seed_fraction, num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="phi", xlabel="WTM Threshold (φ)")


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

    if "switch_probability" in df.columns:
        fig3 = px.line(df, x=x, y="switch_probability", markers=True,
                       title=f"Switch Probability vs {xlabel}",
                       labels={x: xlabel, "switch_probability": "Switch Probability"})
        st.plotly_chart(fig3, use_container_width=True)
