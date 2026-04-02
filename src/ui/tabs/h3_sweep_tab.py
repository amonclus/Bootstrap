from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.h3_parameter_sweep import h3_sweep_beta, h3_sweep_seed_fraction
from ui.state import SidebarConfig


def render_h3_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Transmission rate (β)"],
        key="h3_sweep_type",
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="h3_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="h3_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="h3_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="h3_frac_steps")
        fracs = [
            frac_min + i * (frac_max - frac_min) / (frac_steps - 1)
            for i in range(int(frac_steps))
        ]

        if st.button("▶ Run sweep", key="h3_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = h3_sweep_seed_fraction(
                    graph, fracs,
                    beta=config.beta,
                    gamma=config.gamma,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            _render_charts(pd.DataFrame(data), x="seed_fraction", xlabel="Seed Fraction")

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="h3_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.99, 0.01, key="h3_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 15, key="h3_beta_steps")
        betas = [
            beta_min + i * (beta_max - beta_min) / (beta_steps - 1)
            for i in range(int(beta_steps))
        ]

        if st.button("▶ Run sweep", key="h3_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = h3_sweep_beta(
                    graph, betas,
                    gamma=config.gamma,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                    strategy=config.seed_strategy,
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            fig = px.line(
                df, x="beta", y="cascade_probability", markers=True,
                title="Cascade Probability vs β — transition from SIR-like to bootstrap-like",
                labels={"beta": "β (per-neighbour rate)", "cascade_probability": "Cascade Probability"},
            )
            # Annotate soft threshold region
            fig.add_vline(
                x=1.0 / max(1, graph.number_of_edges() // max(1, graph.number_of_nodes())),
                line_dash="dot", line_color="gray",
                annotation_text="approx. β* (avg degree)",
                annotation_position="top left",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(
                df, x="beta", y="cascade_size", markers=True,
                title="Cascade Size vs β",
                labels={"beta": "β (per-neighbour rate)", "cascade_size": "Avg Cascade Fraction"},
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Show soft threshold m* = 1/β as a secondary axis annotation
            st.caption(
                "The soft infection threshold m\* = 1/β is the number of infected neighbours "
                "required to guarantee infection. As β increases, m\* decreases and the model "
                "approaches bootstrap-like behaviour."
            )
            st.dataframe(
                df[["beta", "soft_threshold_m_star", "cascade_probability", "cascade_size"]]
                .rename(columns={"soft_threshold_m_star": "m* = 1/β"}),
                use_container_width=True,
            )


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
