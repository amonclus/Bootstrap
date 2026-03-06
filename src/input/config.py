from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Config for the simulation."""
    graph_type: str
    num_nodes: int
    p_edge: float | None = None
    radius: float | None = None
    k_threshold: int = 2
    initial_activation_prob: float = 0.1