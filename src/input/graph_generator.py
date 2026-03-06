import networkx as nx


def generate_er_graph(n: int, p: float):
    return nx.erdos_renyi_graph(n, p)


def generate_random_geometric_graph(n: int, radius: float):
    return nx.random_geometric_graph(n, radius)


def generate_lattice_graph(size: int):
    return nx.grid_2d_graph(size, size)