import random


def random_seeds(G, k):
    return random.sample(list(G.nodes()), k)