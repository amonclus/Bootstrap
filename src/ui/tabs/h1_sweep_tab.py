from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.h1_parameter_sweep import h1_sweep_beta, h1_sweep_seed_fraction, h1_sweep_threshold
from ui.state import SidebarConfig


def render_h1_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Transmission rate (β)", "Bootstrap threshold (k)"],
        key="h1_sweep_type",
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="h1_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="h1_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="h1_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="h1_frac_steps")
        fracs = [
            frac_min + i * (frac_max - frac_min) / (frac_steps - 1)
            for i in range(int(frac_steps))
        ]

        if st.button("▶ Run sweep", key="h1_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = h1_sweep_seed_fraction(
                    graph,
                    fracs,
                    threshold=config.threshold,
                    beta=config.beta,
                    gamma=config.gamma,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_fraction_charts(pd.DataFrame(data), x="seed_fraction",
                                    xlabel="Seed Fraction")

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="h1_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.8, 0.05, key="h1_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 10, key="h1_beta_steps")
        betas = [
            beta_min + i * (beta_max - beta_min) / (beta_steps - 1)
            for i in range(int(beta_steps))
        ]

        if st.button("▶ Run sweep", key="h1_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = h1_sweep_beta(
                    graph,
                    betas,
                    threshold=config.threshold,
                    gamma=config.gamma,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_fraction_charts(pd.DataFrame(data), x="beta",
                                    xlabel="Transmission Rate (β)")

    elif sweep_type == "Bootstrap threshold (k)":
        k_min = st.number_input("Min k", 1, 20, 1, key="h1_k_min")
        k_max = st.number_input("Max k", 1, 50, 10, key="h1_k_max")
        thresholds = list(range(int(k_min), int(k_max) + 1))

        if st.button("▶ Run sweep", key="h1_sweep_k_run"):
            with st.spinner("Sweeping bootstrap thresholds…"):
                data = h1_sweep_threshold(
                    graph,
                    thresholds,
                    beta=config.beta,
                    gamma=config.gamma,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_fraction_charts(pd.DataFrame(data), x="threshold",
                                    xlabel="Bootstrap Threshold (k)")


def _render_fraction_charts(df: pd.DataFrame, x: str, xlabel: str) -> None:
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
