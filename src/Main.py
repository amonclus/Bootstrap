from input.graph_generator import choose_graph_source
from simulation.bootstrap import BootstrapPercolation


def main():
    print("Welcome to the network risk analysis tool!")
    print("Choose graph source:")
    print("1 - Generate graph")
    print("2 - Load graph from file")

    choice = input("Enter choice: ")

    g = choose_graph_source(choice)
    if g is None:
        print(f"Failed to load graph. Exiting.")
        return
    print(f"Graph loaded with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

    threshold = int(input("Enter bootstrap threshold (default 2): ") or "2")
    sim = BootstrapPercolation(g, threshold)

    print("Running bootstrap percolation simulation...")
    metrics = sim.collect_metrics(num_trials=50, seed=42)

    print("Simulation results:")

    # Detect degenerate case where cascades are impossible
    if metrics.critical_seed_size == g.number_of_nodes():
        print("  Network is too sparse for cascades with the chosen threshold.")
        print("  No bootstrap cascade can propagate in this graph.")
        print(f"  Critical seed size:           {metrics.critical_seed_size} (all nodes)")
        print("  Cascade probability:          0.0000")
        print("  Time to cascade (avg rounds): 0.00")
        print("  Percolation threshold:        1.0000")
    else:
        print(f"  Cascade size (avg fraction):  {metrics.cascade_size:.4f}")
        print(f"  Critical seed size:           {metrics.critical_seed_size}")
        print(f"  Cascade probability:          {metrics.cascade_probability:.4f}")
        print(f"  Time to cascade (avg rounds): {metrics.time_to_cascade:.2f}")
        print(f"  Percolation threshold:        {metrics.percolation_threshold:.4f}")

if __name__ == "__main__":
    main()