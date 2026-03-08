
from __future__ import annotations

from typing import Dict
import networkx as nx


def compute_graph_statistics(graph: nx.Graph) -> Dict:

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n == 0:
        return {}

    stats = {"nodes": n, "edges": m, "density": nx.density(graph)}

    # Density

    # Degree statistics
    degrees = [deg for _, deg in graph.degree()]
    stats["average_degree"] = sum(degrees) / n
    stats["max_degree"] = max(degrees)
    stats["min_degree"] = min(degrees)

    # Clustering
    stats["average_clustering"] = nx.average_clustering(graph)

    # Connected components
    components = list(nx.connected_components(graph))
    stats["num_components"] = len(components)
    stats["largest_component_size"] = max(len(c) for c in components)

    # Only compute path metrics on largest connected component
    largest_component = graph.subgraph(max(components, key=len))

    if largest_component.number_of_nodes() > 1:
        stats["average_path_length"] = nx.average_shortest_path_length(largest_component)
        stats["diameter"] = nx.diameter(largest_component)
    else:
        stats["average_path_length"] = 0
        stats["diameter"] = 0

    return stats


def degree_distribution(graph: nx.Graph) -> Dict[int, int]:
    distribution = {}

    for _, degree in graph.degree():
        distribution[degree] = distribution.get(degree, 0) + 1

    return distribution


def print_graph_statistics(graph: nx.Graph) -> None:
    stats = compute_graph_statistics(graph)

    print("Graph statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Average degree: {stats['average_degree']:.2f}")
    print(f"  Min degree: {stats['min_degree']}")
    print(f"  Max degree: {stats['max_degree']}")
    print(f"  Average clustering: {stats['average_clustering']:.4f}")
    print(f"  Connected components: {stats['num_components']}")
    print(f"  Largest component size: {stats['largest_component_size']}")
    print(f"  Average path length: {stats['average_path_length']:.2f}")
    print(f"  Diameter: {stats['diameter']}")