import random


def random_seeds(g, k):
    """
    Selects a random sample of k nodes from the graph G to serve as initial seeds for the cascade simulation.
    Args:
        g: nx.Grapg
        k: int

    Returns:
        List of k randomly selected nodes from the graph G
    """
    return random.sample(list(g.nodes()), k)