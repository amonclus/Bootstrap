from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.h2_parameter_sweep import h2_sweep_beta, h2_sweep_seed_fraction, h2_sweep_switch_fraction
from ui.state import SidebarConfig


def render_h2_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Switch threshold (f)", "Transmission rate (β)"],
        key="h2_sweep_type",
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="h2_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="h2_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="h2_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="h2_frac_steps")
        fracs = [
            frac_min + i * (frac_max - frac_min) / (frac_steps - 1)
            for i in range(int(frac_steps))
        ]

        if st.button("▶ Run sweep", key="h2_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = h2_sweep_seed_fraction(
                    graph,
                    fracs,
                    threshold=config.threshold,
                    beta=config.beta,
                    gamma=config.gamma,
                    switch_fraction=config.switch_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="seed_fraction", xlabel="Seed Fraction")

    elif sweep_type == "Switch threshold (f)":
        f_min = st.slider("Min f", 0.01, 0.5, 0.05, 0.01, key="h2_f_min")
        f_max = st.slider("Max f", 0.1, 1.0, 0.8, 0.05, key="h2_f_max")
        f_steps = st.number_input("Number of steps", 3, 30, 10, key="h2_f_steps")
        switch_fracs = [
            f_min + i * (f_max - f_min) / (f_steps - 1)
            for i in range(int(f_steps))
        ]

        if st.button("▶ Run sweep", key="h2_sweep_f_run"):
            with st.spinner("Sweeping switch thresholds…"):
                data = h2_sweep_switch_fraction(
                    graph,
                    switch_fracs,
                    threshold=config.threshold,
                    beta=config.beta,
                    gamma=config.gamma,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            df = pd.DataFrame(data)
            _render_charts(df, x="switch_fraction", xlabel="Switch Threshold (f)")

            # Extra: show how often the switch actually fires vs f
            fig_sw = px.line(
                df,
                x="switch_fraction",
                y="switch_probability",
                markers=True,
                title="Switch Probability vs f — how often does the regime change trigger?",
                labels={"switch_fraction": "Switch Threshold (f)", "switch_probability": "Switch Probability"},
            )
            st.plotly_chart(fig_sw, use_container_width=True)

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="h2_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.8, 0.05, key="h2_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 10, key="h2_beta_steps")
        betas = [
            beta_min + i * (beta_max - beta_min) / (beta_steps - 1)
            for i in range(int(beta_steps))
        ]

        if st.button("▶ Run sweep", key="h2_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = h2_sweep_beta(
                    graph,
                    betas,
                    threshold=config.threshold,
                    gamma=config.gamma,
                    switch_fraction=config.switch_fraction,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="beta", xlabel="Transmission Rate (β)")


def _render_charts(df: pd.DataFrame, x: str, xlabel: str) -> None:
    st.dataframe(df, use_container_width=True)

    fig = px.line(
        df, x=x, y="cascade_probability", markers=True,
        title=f"Cascade Probability vs {xlabel}",
        labels={x: xlabel, "cascade_probability": "Cascade Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df, x=x, y="cascade_size", markers=True,
        title=f"Cascade Size vs {xlabel}",
        labels={x: xlabel, "cascade_size": "Avg Cascade Fraction"},
    )
    st.plotly_chart(fig2, use_container_width=True)
