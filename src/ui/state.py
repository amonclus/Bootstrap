from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import streamlit as st

from simulation.seed_selection import SeedStrategy


class SessionKeys:
    WELCOMED = "welcomed"
    MODEL = "model"
    GRAPH = "graph"
    # Bootstrap percolation
    SIM_RESULTS = "sim_results"
    VULN_DATA = "vuln_data"
    BLOCK_DATA = "block_data"
    BLOCK_BASELINE = "block_baseline"
    # SIR model
    SIR_SIM_RESULTS = "sir_sim_results"
    SIR_VULN_DATA = "sir_vuln_data"
    SIR_BLOCK_DATA = "sir_block_data"
    SIR_BLOCK_BASELINE = "sir_block_baseline"
    # H1 hybrid model
    H1_SIM_RESULTS = "h1_sim_results"
    H1_VULN_DATA = "h1_vuln_data"
    H1_BLOCK_DATA = "h1_block_data"
    H1_BLOCK_BASELINE = "h1_block_baseline"
    # H2 hybrid model
    H2_SIM_RESULTS = "h2_sim_results"
    H2_VULN_DATA = "h2_vuln_data"
    H2_BLOCK_DATA = "h2_block_data"
    H2_BLOCK_BASELINE = "h2_block_baseline"
    # H3 hybrid model
    H3_SIM_RESULTS = "h3_sim_results"
    H3_VULN_DATA = "h3_vuln_data"
    H3_BLOCK_DATA = "h3_block_data"
    H3_BLOCK_BASELINE = "h3_block_baseline"
    # SIS model
    SIS_SIM_RESULTS = "sis_sim_results"
    SIS_VULN_DATA = "sis_vuln_data"
    SIS_BLOCK_DATA = "sis_block_data"
    SIS_BLOCK_BASELINE = "sis_block_baseline"
    # WTM model
    WTM_SIM_RESULTS = "wtm_sim_results"
    WTM_VULN_DATA = "wtm_vuln_data"
    WTM_BLOCK_DATA = "wtm_block_data"
    WTM_BLOCK_BASELINE = "wtm_block_baseline"
    # H4 hybrid model
    H4_SIM_RESULTS = "h4_sim_results"
    H4_VULN_DATA = "h4_vuln_data"
    H4_BLOCK_DATA = "h4_block_data"
    H4_BLOCK_BASELINE = "h4_block_baseline"
    # H5 hybrid model
    H5_SIM_RESULTS = "h5_sim_results"
    H5_VULN_DATA = "h5_vuln_data"
    H5_BLOCK_DATA = "h5_block_data"
    H5_BLOCK_BASELINE = "h5_block_baseline"
    # H6 hybrid model
    H6_SIM_RESULTS = "h6_sim_results"
    H6_VULN_DATA = "h6_vuln_data"
    H6_BLOCK_DATA = "h6_block_data"
    H6_BLOCK_BASELINE = "h6_block_baseline"


@dataclass(frozen=True)
class SidebarConfig:
    threshold: int
    seed_fraction: float
    num_trials: int
    beta: float = 0.3
    gamma: float = 0.1
    seed_strategy: str = SeedStrategy.RANDOM
    switch_fraction: float = 0.2  # H2/H5: fraction of population infected before switching phases
    phi: float = 0.3              # WTM/H4/H5/H6: fractional neighbour threshold


@dataclass(frozen=True)
class SimulationContext:
    threshold: int
    seed_fraction: float
    num_trials: int

    @property
    def seed_size(self) -> int:
        graph = get_graph_or_stop()
        return max(1, int(self.seed_fraction * graph.number_of_nodes()))


def get_graph_or_stop() -> nx.Graph:
    graph = st.session_state.get(SessionKeys.GRAPH)
    if graph is None:
        st.info("👈 Configure and generate (or upload) a graph in the sidebar to get started.")
        st.stop()
    return graph


def clear_sim_results() -> None:
    """Remove all cached simulation results from session state."""
    result_keys = [
        SessionKeys.SIM_RESULTS, SessionKeys.VULN_DATA,
        SessionKeys.BLOCK_DATA, SessionKeys.BLOCK_BASELINE,
        SessionKeys.SIR_SIM_RESULTS, SessionKeys.SIR_VULN_DATA,
        SessionKeys.SIR_BLOCK_DATA, SessionKeys.SIR_BLOCK_BASELINE,
        SessionKeys.H1_SIM_RESULTS, SessionKeys.H1_VULN_DATA,
        SessionKeys.H1_BLOCK_DATA, SessionKeys.H1_BLOCK_BASELINE,
        SessionKeys.H2_SIM_RESULTS, SessionKeys.H2_VULN_DATA,
        SessionKeys.H2_BLOCK_DATA, SessionKeys.H2_BLOCK_BASELINE,
        SessionKeys.H3_SIM_RESULTS, SessionKeys.H3_VULN_DATA,
        SessionKeys.H3_BLOCK_DATA, SessionKeys.H3_BLOCK_BASELINE,
        SessionKeys.SIS_SIM_RESULTS, SessionKeys.SIS_VULN_DATA,
        SessionKeys.SIS_BLOCK_DATA, SessionKeys.SIS_BLOCK_BASELINE,
        SessionKeys.WTM_SIM_RESULTS, SessionKeys.WTM_VULN_DATA,
        SessionKeys.WTM_BLOCK_DATA, SessionKeys.WTM_BLOCK_BASELINE,
        SessionKeys.H4_SIM_RESULTS, SessionKeys.H4_VULN_DATA,
        SessionKeys.H4_BLOCK_DATA, SessionKeys.H4_BLOCK_BASELINE,
        SessionKeys.H5_SIM_RESULTS, SessionKeys.H5_VULN_DATA,
        SessionKeys.H5_BLOCK_DATA, SessionKeys.H5_BLOCK_BASELINE,
        SessionKeys.H6_SIM_RESULTS, SessionKeys.H6_VULN_DATA,
        SessionKeys.H6_BLOCK_DATA, SessionKeys.H6_BLOCK_BASELINE,
    ]
    for key in result_keys:
        st.session_state.pop(key, None)

