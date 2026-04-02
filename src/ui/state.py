from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import streamlit as st

from simulation.seed_selection import SeedStrategy


class SessionKeys:
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


@dataclass(frozen=True)
class SidebarConfig:
    threshold: int
    seed_fraction: float
    num_trials: int
    beta: float = 0.3
    gamma: float = 0.1
    seed_strategy: str = SeedStrategy.RANDOM
    switch_fraction: float = 0.2  # H2: fraction of population infected before switching phases


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

