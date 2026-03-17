def write_graph(graph, output_file):
    """
    Writes a graph in dimacs format.
    Args:
        graph: graph in dimacs format
        output_file: path to output file
    """
    with open(output_file, "w") as f:
        nodes = list(graph.nodes())
        node_to_id = {node: i+1 for i, node in enumerate(nodes)}

        f.write(f"p edge {len(nodes)} {graph.number_of_edges()}\n")

        for u, v in graph.edges():
            f.write(f"e {node_to_id[u]} {node_to_id[v]}\n")