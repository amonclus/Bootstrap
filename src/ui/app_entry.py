from __future__ import annotations

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import SessionKeys, get_graph_or_stop
from ui.tabs.animation_tab import render_animation_tab
from ui.tabs.simulation_tab import render_simulation_tab
from ui.tabs.stats_tab import render_stats_tab
from ui.tabs.sweep_tab import render_sweep_tab
from ui.tabs.vulnerability_tab import render_vulnerability_tab
from ui.tabs.sir_simulation_tab import render_sir_simulation_tab
from ui.tabs.sir_animation_tab import render_sir_animation_tab
from ui.tabs.sir_vulnerability_tab import render_sir_vulnerability_tab
from ui.tabs.sir_sweep_tab import render_sir_sweep_tab
from ui.tabs.h1_simulation_tab import render_h1_simulation_tab
from ui.tabs.h1_animation_tab import render_h1_animation_tab
from ui.tabs.h1_vulnerability_tab import render_h1_vulnerability_tab
from ui.tabs.h1_sweep_tab import render_h1_sweep_tab
from ui.tabs.h2_simulation_tab import render_h2_simulation_tab
from ui.tabs.h2_animation_tab import render_h2_animation_tab
from ui.tabs.h2_vulnerability_tab import render_h2_vulnerability_tab
from ui.tabs.h2_sweep_tab import render_h2_sweep_tab
from ui.tabs.h3_simulation_tab import render_h3_simulation_tab
from ui.tabs.h3_animation_tab import render_h3_animation_tab
from ui.tabs.h3_vulnerability_tab import render_h3_vulnerability_tab
from ui.tabs.h3_sweep_tab import render_h3_sweep_tab


def run_app() -> None:
    st.set_page_config(
        page_title="Network Contagion Lab",
        page_icon="🌐",
        layout="centered",
    )

    if SessionKeys.MODEL not in st.session_state:
        _render_welcome()
        return

    model = st.session_state[SessionKeys.MODEL]

    if model == "bootstrap":
        st.title("Bootstrap Percolation — Network Risk Analysis")
        config = render_sidebar(model="bootstrap")
        graph = get_graph_or_stop()
        _render_tabs_bootstrap(graph, config)
    elif model == "sir":
        st.title("SIR Epidemic Model — Network Spread Analysis")
        config = render_sidebar(model="sir")
        graph = get_graph_or_stop()
        _render_tabs_sir(graph, config)
    elif model == "h1":
        st.title("H1 — OR-Hybrid Contagion Model")
        config = render_sidebar(model="h1")
        graph = get_graph_or_stop()
        _render_tabs_h1(graph, config)
    elif model == "h2":
        st.title("H2 — Sequential Hybrid (Switching Model)")
        config = render_sidebar(model="h2")
        graph = get_graph_or_stop()
        _render_tabs_h2(graph, config)
    elif model == "h3":
        st.title("H3 — Probabilistic Threshold Hybrid")
        config = render_sidebar(model="h3")
        graph = get_graph_or_stop()
        _render_tabs_h3(graph, config)


def _render_welcome() -> None:
    st.title("🌐 Network Contagion Lab")
    st.markdown(
        "A research tool for studying how things spread through networks — "
        "failures, diseases, information. Choose a model below to get started."
    )

    # ── Base models ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Base Models")

    col_bp, col_sir = st.columns(2)

    with col_bp:
        st.markdown("### Cascade / Bootstrap Percolation")
        st.markdown(
            "Models **threshold-driven** failures. A node fails as soon as it has "
            "at least **k** failed neighbours. The cascade is deterministic given "
            "the seed set.\n\n"
            "**Use this to study:** infrastructure failures, opinion tipping points, "
            "or any system where nodes fail collectively once enough neighbours fail."
        )
        st.markdown("**Key parameter:** k — failure threshold")
        if st.button("Use Bootstrap Percolation →", use_container_width=True):
            st.session_state[SessionKeys.MODEL] = "bootstrap"
            st.rerun()

    with col_sir:
        st.markdown("### SIR Epidemic Model")
        st.markdown(
            "Models **probabilistic** contagion with recovery. Each infected node "
            "tries to infect each susceptible neighbour with probability **β** every "
            "round, and recovers with probability **γ**. Spread is stochastic.\n\n"
            "**Use this to study:** disease outbreaks, virus propagation, "
            "or any system where spread is noisy and recovery is possible."
        )
        st.markdown("**Key parameters:** β — transmission rate · γ — recovery rate")
        if st.button("Use SIR Model →", use_container_width=True):
            st.session_state[SessionKeys.MODEL] = "sir"
            st.rerun()

    # ── Hybrid models ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Hybrid Models")
    st.caption(
        "Hybrid models combine the SIR and bootstrap percolation channels. "
        "Each variant defines a different rule for how the two channels interact."
    )

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.markdown("### H1 — OR Hybrid")
        st.markdown(
            "A node is infected if **either** the SIR channel fires (β) **or** the "
            "bootstrap threshold is met (k), whichever comes first. Recovery at rate γ."
        )
        st.markdown("**Parameters:** k · β · γ")
        if st.button("Use H1 →", use_container_width=True):
            st.session_state[SessionKeys.MODEL] = "h1"
            st.rerun()

    with h_col2:
        st.markdown("### H2 — Sequential Hybrid")
        st.markdown(
            "Runs **SIR** until a fraction **f** of the population is ever-infected, "
            "then switches to **bootstrap percolation** for the remainder. "
            "Models a behavioural shift as the outbreak becomes visible."
        )
        st.markdown("**Parameters:** k · β · γ · f")
        if st.button("Use H2 →", use_container_width=True):
            st.session_state[SessionKeys.MODEL] = "h2"
            st.rerun()

    with h_col3:
        st.markdown("### H3 — Probabilistic Threshold")
        st.markdown(
            "A node with **m** infected neighbours is infected with probability **m·β**. "
            "Each neighbour adds an independent contribution — soft reinforcement without "
            "a hard threshold. The natural threshold m\* = 1/β emerges from β."
        )
        st.markdown("**Parameters:** β · γ")
        if st.button("Use H3 →", use_container_width=True):
            st.session_state[SessionKeys.MODEL] = "h3"
            st.rerun()

    h_col4, h_col5, h_col6 = st.columns(3)

    with h_col4:
        st.markdown("### H4")
        st.markdown("*Coming soon.*")
        st.button("H4 (coming soon)", disabled=True, use_container_width=True)

    with h_col5:
        st.markdown("### H5")
        st.markdown("*Coming soon.*")
        st.button("H5 (coming soon)", disabled=True, use_container_width=True)

    with h_col6:
        st.markdown("### H6")
        st.markdown("*Coming soon.*")
        st.button("H6 (coming soon)", disabled=True, use_container_width=True)

    st.markdown("---")
    st.caption(
        "All models run on the same network types: Erdős–Rényi, Random Geometric, "
        "and Lattice graphs, or any graph you upload (DIMACS, edge list, GML)."
    )


def _render_tabs_bootstrap(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_simulation_tab(graph, config)

    with tab_anim:
        render_animation_tab(graph, config)

    with tab_vuln:
        render_vulnerability_tab(graph, config)

    with tab_sweep:
        render_sweep_tab(graph, config)


def _render_tabs_sir(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Epidemic Simulation",
            "🎬 Epidemic Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_sir_simulation_tab(graph, config)

    with tab_anim:
        render_sir_animation_tab(graph, config)

    with tab_vuln:
        render_sir_vulnerability_tab(graph, config)

    with tab_sweep:
        render_sir_sweep_tab(graph, config)


def _render_tabs_h3(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h3_simulation_tab(graph, config)

    with tab_anim:
        render_h3_animation_tab(graph, config)

    with tab_vuln:
        render_h3_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h3_sweep_tab(graph, config)


def _render_tabs_h2(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h2_simulation_tab(graph, config)

    with tab_anim:
        render_h2_animation_tab(graph, config)

    with tab_vuln:
        render_h2_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h2_sweep_tab(graph, config)


def _render_tabs_h1(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h1_simulation_tab(graph, config)

    with tab_anim:
        render_h1_animation_tab(graph, config)

    with tab_vuln:
        render_h1_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h1_sweep_tab(graph, config)
