from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.parameter_sweep import (
    sweep_er_probability,
    sweep_geometric_radius,
    sweep_lattice_size,
    sweep_seed_fraction,
)
from ui.state import SidebarConfig


def render_sweep_tab(graph: nx.Graph, config: SidebarConfig) -> None:
    st.subheader("Parameter Sweep")

    sweep_type = st.selectbox(
        "Sweep type",
        ["Seed Fraction", "Erdős–Rényi probability", "Geometric radius", "Lattice size"],
    )

    sweep_trials = st.number_input("Trials per point", 10, 200, 30, step=10, key="sweep_trials")

    if sweep_type == "Seed Fraction":
        frac_min = st.slider("Min fraction", 0.01, 0.5, 0.01, 0.01)
        frac_max = st.slider("Max fraction", 0.1, 1.0, 0.5, 0.05)
        frac_steps = st.number_input("Number of steps", 3, 30, 10)
        fracs = [
            frac_min + i * (frac_max - frac_min) / (frac_steps - 1)
            for i in range(int(frac_steps))
        ]

        if st.button("▶ Run sweep", key="sweep_run"):
            with st.spinner("Running seed-fraction sweep…"):
                data = sweep_seed_fraction(
                    graph,
                    fracs,
                    threshold=config.threshold,
                    num_trials=sweep_trials,
                    strategy=config.seed_strategy,
                )
            _render_seed_fraction_charts(pd.DataFrame(data))

    elif sweep_type == "Erdős–Rényi probability":
        er_n = st.number_input("Number of nodes", 10, 2000, 100, key="er_n")
        er_k = st.number_input("Threshold", 1, 20, 2, key="er_k")
        probs = st.text_input("Probabilities (comma-separated)", "0.01,0.05,0.1,0.2,0.3")
        prob_list = [float(x.strip()) for x in probs.split(",")]

        if st.button("▶ Run sweep", key="sweep_er"):
            with st.spinner("Sweeping ER probabilities…"):
                data = sweep_er_probability(er_n, prob_list, threshold=er_k, num_trials=sweep_trials, strategy=config.seed_strategy)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(df, x="p", y="cascade_probability", markers=True, title="Cascade Probability vs p")
            st.plotly_chart(fig, use_container_width=True)

    elif sweep_type == "Geometric radius":
        geo_n = st.number_input("Number of nodes", 10, 2000, 100, key="geo_n")
        geo_k = st.number_input("Threshold", 1, 20, 2, key="geo_k")
        radii_str = st.text_input("Radii (comma-separated)", "0.05,0.1,0.15,0.2,0.25,0.3")
        radii_list = [float(x.strip()) for x in radii_str.split(",")]

        if st.button("▶ Run sweep", key="sweep_geo"):
            with st.spinner("Sweeping geometric radii…"):
                data = sweep_geometric_radius(geo_n, radii_list, threshold=geo_k, num_trials=sweep_trials, strategy=config.seed_strategy)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(df, x="radius", y="cascade_probability", markers=True, title="Cascade Probability vs Radius")
            st.plotly_chart(fig, use_container_width=True)

    elif sweep_type == "Lattice size":
        lat_k = st.number_input("Threshold", 1, 20, 2, key="lat_k")
        sizes_str = st.text_input("Grid sizes (comma-separated)", "5,10,15,20")
        sizes_list = [int(x.strip()) for x in sizes_str.split(",")]

        if st.button("▶ Run sweep", key="sweep_lat"):
            with st.spinner("Sweeping lattice sizes…"):
                data = sweep_lattice_size(sizes_list, threshold=lat_k, num_trials=sweep_trials, strategy=config.seed_strategy)
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            fig = px.line(df, x="grid_size", y="cascade_probability", markers=True, title="Cascade Probability vs Grid Size")
            st.plotly_chart(fig, use_container_width=True)


def _render_seed_fraction_charts(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)

    fig = px.line(
        df,
        x="seed_fraction",
        y="cascade_probability",
        markers=True,
        title="Cascade Probability vs Seed Fraction",
        labels={"seed_fraction": "Seed Fraction", "cascade_probability": "Cascade Probability"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="seed_fraction",
        y="cascade_size",
        markers=True,
        title="Cascade Size vs Seed Fraction",
        labels={"seed_fraction": "Seed Fraction", "cascade_size": "Avg Cascade Fraction"},
    )
    st.plotly_chart(fig2, use_container_width=True)

