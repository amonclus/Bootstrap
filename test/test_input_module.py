from src.input.graph_generator import (
    generate_er_graph,
    generate_random_geometric_graph,
    generate_lattice_graph
)

from src.input.graph_loader import (
    load_graph_from_dimacs,
    load_graph_from_edge_list,
    load_graph_from_gml
)


def main():


    print("Choose graph source:")
    print("1 - Generate graph")
    print("2 - Load graph from file")

    choice = input("Selection: ")

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
            return

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
            return

    else:
        print("Invalid selection")
        return

    print("\nGraph successfully created!")
    print("Number of nodes:", g.number_of_nodes())
    print("Number of edges:", g.number_of_edges())


if __name__ == "__main__":
    main()