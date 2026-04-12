"""
Reads a file containing a graph in various formats and loads it into a NetworkX graph object.
"""
import networkx as nx


def load_graph_from_dimacs(path: str):
    """
    Loads a graph in dimacs format.
    Args:
        path: path to the dimacs file
    """
    g = nx.Graph()

    with open(path, "r") as f:
        for line in f:
            if line.startswith("e"):
                _, u, v = line.split()
                g.add_edge(int(u), int(v))

    return g

def load_graph_from_edge_list(path: str):
    """
    Loads a graph in edge list.
    Args:
        path: path to the edge list file
    """
    g = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])
    return g

def load_graph_from_gml(path: str):
    """
    Loads a graph in gml format.
    Args:
        path: path to the gml file
    """
    return nx.read_gml(path)