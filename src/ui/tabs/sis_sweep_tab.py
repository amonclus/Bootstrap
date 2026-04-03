from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.sis_parameter_sweep import sis_sweep_beta, sis_sweep_gamma, sis_sweep_seed_fraction
from ui.state import SidebarConfig


def render_sis_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Transmission rate (β)", "Recovery rate (μ)"],
        key="sis_sweep_type",
    )
    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="sis_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="sis_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="sis_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="sis_frac_steps")
        fracs = [frac_min + i * (frac_max - frac_min) / (frac_steps - 1) for i in range(int(frac_steps))]

        if st.button("▶ Run sweep", key="sis_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = sis_sweep_seed_fraction(
                    graph, fracs, beta=config.beta, gamma=config.gamma,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="seed_fraction", xlabel="Seed Fraction")

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="sis_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.8, 0.05, key="sis_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 10, key="sis_beta_steps")
        betas = [beta_min + i * (beta_max - beta_min) / (beta_steps - 1) for i in range(int(beta_steps))]

        if st.button("▶ Run sweep", key="sis_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = sis_sweep_beta(
                    graph, betas, gamma=config.gamma, seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="beta", xlabel="Transmission Rate (β)")

    elif sweep_type == "Recovery rate (μ)":
        g_min = st.slider("Min μ", 0.01, 0.5, 0.01, 0.01, key="sis_g_min")
        g_max = st.slider("Max μ", 0.1, 1.0, 0.8, 0.05, key="sis_g_max")
        g_steps = st.number_input("Number of steps", 3, 30, 10, key="sis_g_steps")
        gammas = [g_min + i * (g_max - g_min) / (g_steps - 1) for i in range(int(g_steps))]

        if st.button("▶ Run sweep", key="sis_sweep_g_run"):
            with st.spinner("Sweeping recovery rates…"):
                data = sis_sweep_gamma(
                    graph, gammas, beta=config.beta, seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials), strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="gamma", xlabel="Recovery Rate (μ)")


def _render_charts(df: pd.DataFrame, x: str, xlabel: str) -> None:
    st.dataframe(df, use_container_width=True)

    fig = px.line(df, x=x, y="cascade_probability", markers=True,
                  title=f"Epidemic Probability vs {xlabel}",
                  labels={x: xlabel, "cascade_probability": "Epidemic Probability"})
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(df, x=x, y="cascade_size", markers=True,
                   title=f"Peak Epidemic Size vs {xlabel}",
                   labels={x: xlabel, "cascade_size": "Avg Peak Fraction"})
    st.plotly_chart(fig2, use_container_width=True)
