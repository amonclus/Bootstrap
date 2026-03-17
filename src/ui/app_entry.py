from __future__ import annotations

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import get_graph_or_stop
from ui.tabs.animation_tab import render_animation_tab
from ui.tabs.simulation_tab import render_simulation_tab
from ui.tabs.stats_tab import render_stats_tab
from ui.tabs.sweep_tab import render_sweep_tab
from ui.tabs.vulnerability_tab import render_vulnerability_tab


def run_app() -> None:
    st.set_page_config(
        page_title="Bootstrap Percolation - Network Risk Analysis",
        page_icon="🌐",
        layout="centered",
    )

    st.title("🌐 Bootstrap Percolation - Network Risk Analysis")
    st.caption("A showcase tool for analysing cascade risk in networks.")

    config = render_sidebar()
    graph = get_graph_or_stop()

    tab_stats, tab_sim, tab_vuln, tab_anim, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎯 Node Vulnerability",
            "🎬 Cascade Animation",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_simulation_tab(graph, config)

    with tab_vuln:
        render_vulnerability_tab(graph, config)

    with tab_anim:
        render_animation_tab(graph, config)

    with tab_sweep:
        render_sweep_tab(graph, config)

