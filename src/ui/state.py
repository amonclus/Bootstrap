from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import streamlit as st


class SessionKeys:
    GRAPH = "graph"
    SIM_RESULTS = "sim_results"
    VULN_DATA = "vuln_data"
    BLOCK_DATA = "block_data"
    BLOCK_BASELINE = "block_baseline"


@dataclass(frozen=True)
class SidebarConfig:
    threshold: int
    seed_fraction: float
    num_trials: int


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

