from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.sir_parameter_sweep import (
    sir_sweep_beta,
    sir_sweep_er_probability,
    sir_sweep_lattice_size,
    sir_sweep_seed_fraction,
)
from ui.state import SidebarConfig


def render_sir_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Transmission rate (β)", "Erdős–Rényi probability", "Lattice size"],
        key="sir_sweep_type",
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="sir_sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01, key="sir_frac_min")
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05, key="sir_frac_max")
        frac_steps = st.number_input("Number of steps", 3, 30, 10, key="sir_frac_steps")
        fracs = [
            frac_min + i * (frac_max - frac_min) / (frac_steps - 1)
            for i in range(int(frac_steps))
        ]

        if st.button("▶ Run sweep", key="sir_sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = sir_sweep_seed_fraction(
                    graph,
                    fracs,
                    beta=config.beta,
                    gamma=config.gamma,
                    num_trials=int(sweep_trials),
                )
            _render_seed_fraction_charts(pd.DataFrame(data))

    elif sweep_type == "Transmission rate (β)":
        beta_min = st.slider("Min β", 0.01, 0.5, 0.05, 0.01, key="sir_beta_min")
        beta_max = st.slider("Max β", 0.1, 1.0, 0.8, 0.05, key="sir_beta_max")
        beta_steps = st.number_input("Number of steps", 3, 30, 10, key="sir_beta_steps")
        betas = [
            beta_min + i * (beta_max - beta_min) / (beta_steps - 1)
            for i in range(int(beta_steps))
        ]

        if st.button("▶ Run sweep", key="sir_sweep_beta_run"):
            with st.spinner("Sweeping transmission rates…"):
                data = sir_sweep_beta(
                    graph,
                    betas,
                    gamma=config.gamma,
                    seed_fraction=config.seed_fraction,
                    num_trials=int(sweep_trials),
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(
                df, x="beta", y="epidemic_probability", markers=True,
                title="Epidemic Probability vs β",
                labels={"beta": "Transmission Rate (β)", "epidemic_probability": "Epidemic Probability"},
            )
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(
                df, x="beta", y="epidemic_size", markers=True,
                title="Epidemic Size vs β",
                labels={"beta": "Transmission Rate (β)", "epidemic_size": "Avg Epidemic Fraction"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif sweep_type == "Erdős–Rényi probability":
        er_n = st.number_input("Number of nodes", 10, 2000, 100, key="sir_er_n")
        probs = st.text_input("Probabilities (comma-separated)", "0.01,0.05,0.1,0.2,0.3", key="sir_er_probs")
        prob_list = [float(x.strip()) for x in probs.split(",")]

        if st.button("▶ Run sweep", key="sir_sweep_er"):
            with st.spinner("Sweeping ER probabilities…"):
                data = sir_sweep_er_probability(
                    int(er_n), prob_list,
                    beta=config.beta,
                    gamma=config.gamma,
                    num_trials=int(sweep_trials),
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(df, x="p", y="epidemic_probability", markers=True,
                          title="Epidemic Probability vs p")
            st.plotly_chart(fig, use_container_width=True)

    elif sweep_type == "Lattice size":
        sizes_str = st.text_input("Grid sizes (comma-separated)", "5,10,15,20", key="sir_lat_sizes")
        sizes_list = [int(x.strip()) for x in sizes_str.split(",")]

        if st.button("▶ Run sweep", key="sir_sweep_lat"):
            with st.spinner("Sweeping lattice sizes…"):
                data = sir_sweep_lattice_size(
                    sizes_list,
                    beta=config.beta,
                    gamma=config.gamma,
                    num_trials=int(sweep_trials),
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(df, x="grid_size", y="epidemic_probability", markers=True,
                          title="Epidemic Probability vs Grid Size")
            st.plotly_chart(fig, use_container_width=True)


def _render_seed_fraction_charts(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)

    fig = px.line(
        df,
        x="seed_fraction",
        y="epidemic_probability",
        markers=True,
        title="Epidemic Probability vs Seed Fraction",
        labels={"seed_fraction": "Seed Fraction", "epidemic_probability": "Epidemic Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="seed_fraction",
        y="epidemic_size",
        markers=True,
        title="Epidemic Size vs Seed Fraction",
        labels={"seed_fraction": "Seed Fraction", "epidemic_size": "Avg Epidemic Fraction"},
    )
    st.plotly_chart(fig2, use_container_width=True)
