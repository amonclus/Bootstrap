import networkx as nx


def load_graph_from_dimacs(path: str):
    g = nx.Graph()

    with open(path, "r") as f:
        for line in f:
            if line.startswith("e"):
                _, u, v = line.split()
                g.add_edge(int(u), int(v))

    return g

def load_graph_from_edge_list(path: str):
    return nx.from_edgelist(nx.read_edgelist(path))

def load_graph_from_gml(path: str):
    return nx.read_gml(path)