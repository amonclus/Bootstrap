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
from ui.tabs.sis_simulation_tab import render_sis_simulation_tab
from ui.tabs.sis_animation_tab import render_sis_animation_tab
from ui.tabs.sis_vulnerability_tab import render_sis_vulnerability_tab
from ui.tabs.sis_sweep_tab import render_sis_sweep_tab
from ui.tabs.wtm_simulation_tab import render_wtm_simulation_tab
from ui.tabs.wtm_animation_tab import render_wtm_animation_tab
from ui.tabs.wtm_vulnerability_tab import render_wtm_vulnerability_tab
from ui.tabs.wtm_sweep_tab import render_wtm_sweep_tab
from ui.tabs.h4_simulation_tab import render_h4_simulation_tab
from ui.tabs.h4_animation_tab import render_h4_animation_tab
from ui.tabs.h4_vulnerability_tab import render_h4_vulnerability_tab
from ui.tabs.h4_sweep_tab import render_h4_sweep_tab
from ui.tabs.h5_simulation_tab import render_h5_simulation_tab
from ui.tabs.h5_animation_tab import render_h5_animation_tab
from ui.tabs.h5_vulnerability_tab import render_h5_vulnerability_tab
from ui.tabs.h5_sweep_tab import render_h5_sweep_tab
from ui.tabs.h6_simulation_tab import render_h6_simulation_tab
from ui.tabs.h6_animation_tab import render_h6_animation_tab
from ui.tabs.h6_vulnerability_tab import render_h6_vulnerability_tab
from ui.tabs.h6_sweep_tab import render_h6_sweep_tab


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
    elif model == "sis":
        st.title("SIS Epidemic Model — Reinfection Dynamics")
        config = render_sidebar(model="sis")
        graph = get_graph_or_stop()
        _render_tabs_sis(graph, config)
    elif model == "wtm":
        st.title("WTM — Watts Threshold Model")
        config = render_sidebar(model="wtm")
        graph = get_graph_or_stop()
        _render_tabs_wtm(graph, config)
    elif model == "h4":
        st.title("H4 — OR-Hybrid: SIS + Watts Threshold Model")
        config = render_sidebar(model="h4")
        graph = get_graph_or_stop()
        _render_tabs_h4(graph, config)
    elif model == "h5":
        st.title("H5 — Sequential Hybrid: SIS then WTM")
        config = render_sidebar(model="h5")
        graph = get_graph_or_stop()
        _render_tabs_h5(graph, config)
    elif model == "h6":
        st.title("H6 — Probabilistic WTM (Soft Threshold)")
        config = render_sidebar(model="h6")
        graph = get_graph_or_stop()
        _render_tabs_h6(graph, config)


def _render_model_row(models: list) -> None:
    """Render a row of model cards with buttons guaranteed to be aligned.

    Descriptions go in one st.columns block; buttons go in a separate
    st.columns block immediately below.  Because both blocks share the
    same column widths, the buttons always start at the same vertical
    position regardless of how tall each description is.
    """
    n = len(models)
    text_cols = st.columns(n)
    for col, (_, title, desc, params, _btn) in zip(text_cols, models):
        with col:
            st.markdown(f"### {title}")
            st.markdown(desc)
            st.markdown(params)

    btn_cols = st.columns(n)
    for col, (model_key, _title, _desc, _params, btn_label) in zip(btn_cols, models):
        with col:
            if st.button(btn_label, use_container_width=True, key=f"btn_{model_key}"):
                st.session_state[SessionKeys.MODEL] = model_key
                st.rerun()


def _render_welcome() -> None:
    st.title("🌐 Network Contagion Lab")
    st.markdown(
        "A research tool for studying how things spread through networks — "
        "failures, diseases, information. Choose a model below to get started."
    )

    # ── Base models ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Base Models")

    _BASE_MODELS = [
        ("bootstrap", "Bootstrap Percolation",
         "**Threshold-driven** cascade. A node fails once ≥ **k** neighbours have failed. "
         "The cascade is fully deterministic given the seed set.",
         "**Key parameter:** k",
         "Use Bootstrap Percolation →"),
        ("sir", "SIR Epidemic Model",
         "**Probabilistic** contagion with **permanent immunity**. Each infected node "
         "tries to spread at rate β; recovery at rate γ. Recovered nodes cannot be re-infected.",
         "**Key parameters:** β · γ",
         "Use SIR →"),
        ("sis", "SIS Epidemic Model",
         "**Probabilistic** contagion **without** permanent immunity. Recovered nodes "
         "return to susceptible and can be re-infected. Endemic equilibria are possible.",
         "**Key parameters:** β · μ",
         "Use SIS →"),
        ("wtm", "Watts Threshold Model",
         "**Fractional** threshold cascade. A node activates once the *fraction* of "
         "infected neighbours ≥ **φ**. Degree-normalised analogue of bootstrap percolation.",
         "**Key parameter:** φ",
         "Use WTM →"),
    ]

    _render_model_row(_BASE_MODELS)

    # ── Hybrid models ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Hybrid Models")
    st.caption(
        "Hybrid models combine epidemic and threshold channels. "
        "Each variant defines a different rule for how the two mechanisms interact."
    )

    _HYBRID_MODELS = [
        ("h1", "H1 — SIR ∨ Bootstrap",
         "Infects if **either** the SIR channel fires (β) **or** the bootstrap "
         "threshold is met (k). Recovery at rate γ → permanently immune.",
         "**Parameters:** k · β · γ",
         "Use H1 →"),
        ("h2", "H2 — SIR → Bootstrap",
         "Runs **SIR** until fraction **f** ever-infected, then switches to "
         "**bootstrap percolation**. Models behavioural change once an outbreak is visible.",
         "**Parameters:** k · β · γ · f",
         "Use H2 →"),
        ("h3", "H3 — Soft Bootstrap",
         "Infection probability scales linearly with infected-neighbour count: "
         "P = min(1, m·β). Soft social reinforcement without a hard threshold.",
         "**Parameters:** β · γ",
         "Use H3 →"),
        ("h4", "H4 — SIS ∨ WTM",
         "Infects if **either** the SIS channel fires (β) **or** the fractional "
         "WTM threshold is met (φ). Recovery at rate γ → susceptible again.",
         "**Parameters:** φ · β · γ",
         "Use H4 →"),
        ("h5", "H5 — SIS → WTM",
         "Runs **SIS** until fraction **f** simultaneously infected, then switches "
         "to **WTM**. Fractional threshold φ governs Phase 2.",
         "**Parameters:** φ · β · γ · f",
         "Use H5 →"),
        ("h6", "H6 — Soft WTM",
         "Infection probability scales with the fraction of infected neighbours "
         "divided by φ: P = min(1, (m/deg)/φ). Soft fractional reinforcement.",
         "**Parameters:** φ · γ",
         "Use H6 →"),
    ]

    _render_model_row(_HYBRID_MODELS[:3])
    _render_model_row(_HYBRID_MODELS[3:])

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


def _render_tabs_sis(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Epidemic Simulation", "🎬 Epidemic Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_sis_simulation_tab(graph, config)
    with tab_anim:
        render_sis_animation_tab(graph, config)
    with tab_vuln:
        render_sis_vulnerability_tab(graph, config)
    with tab_sweep:
        render_sis_sweep_tab(graph, config)


def _render_tabs_wtm(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_wtm_simulation_tab(graph, config)
    with tab_anim:
        render_wtm_animation_tab(graph, config)
    with tab_vuln:
        render_wtm_vulnerability_tab(graph, config)
    with tab_sweep:
        render_wtm_sweep_tab(graph, config)


def _render_tabs_h4(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h4_simulation_tab(graph, config)
    with tab_anim:
        render_h4_animation_tab(graph, config)
    with tab_vuln:
        render_h4_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h4_sweep_tab(graph, config)


def _render_tabs_h5(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h5_simulation_tab(graph, config)
    with tab_anim:
        render_h5_animation_tab(graph, config)
    with tab_vuln:
        render_h5_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h5_sweep_tab(graph, config)


def _render_tabs_h6(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h6_simulation_tab(graph, config)
    with tab_anim:
        render_h6_animation_tab(graph, config)
    with tab_vuln:
        render_h6_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h6_sweep_tab(graph, config)


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
