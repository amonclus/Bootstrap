import networkx as nx
from input.graph_loader import load_graph_from_gml, load_graph_from_edge_list, load_graph_from_dimacs
from input.write_graph import write_graph


def generate_er_graph(n: int, p: float):
    return nx.erdos_renyi_graph(n, p)


def generate_random_geometric_graph(n: int, radius: float):
    return nx.random_geometric_graph(n, radius)


def generate_lattice_graph(size: int):
    return nx.grid_2d_graph(size, size)


def choose_graph_source(choice):
    if choice == "1":
        print("\nChoose graph type:")
        print("1 - Erdős–Rényi")
        print("2 - Random Geometric")
        print("3 - Lattice")

        gtype = input("Selection: ")

        if gtype == "1":
            n = int(input("Number of nodes: "))
            p = float(input("Edge probability: "))
            g = generate_er_graph(n, p)

        elif gtype == "2":
            n = int(input("Number of nodes: "))
            r = float(input("Connection radius: "))
            g = generate_random_geometric_graph(n, r)

        elif gtype == "3":
            size = int(input("Grid size: "))
            g = generate_lattice_graph(size)

        else:
            print("Invalid selection")
            return None

        write_graph(g, "generated_graph.dimacs")
        return g

    elif choice == "2":

        print("\nChoose file format:")
        print("1 - DIMACS")
        print("2 - Edge List")
        print("3 - GML")

        ftype = input("Selection: ")
        path = input("Path to file: ")

        if ftype == "1":
            g = load_graph_from_dimacs(path)

        elif ftype == "2":
            g = load_graph_from_edge_list(path)

        elif ftype == "3":
            g = load_graph_from_gml(path)

        else:
            print("Invalid selection")
            return None

        return g

    else:
        print("Invalid selection")
        return None
