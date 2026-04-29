"""
Reads a file containing a graph in various formats and loads it into a NetworkX graph object.
"""
from __future__ import annotations

import os

import networkx as nx


def load_graph_auto(path: str) -> nx.Graph:
    """Detect the file format and load the graph automatically.

    Detection order:
    1. File extension (.gml, .dimacs, .edgelist)
    2. Content sniffing (looks for GML or DIMACS markers)
    3. Falls back to edge-list parsing
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".gml":
        return load_graph_from_gml(path)
    if ext == ".dimacs":
        return load_graph_from_dimacs(path)
    if ext in {".edgelist"}:
        return load_graph_from_edge_list(path)

    # Extension is ambiguous (.txt, etc.) — sniff the content
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(512)

    if "graph [" in head or head.lstrip().startswith("Creator"):
        return load_graph_from_gml(path)
    if any(line.startswith("p ") or line.startswith("e ") for line in head.splitlines()):
        return load_graph_from_dimacs(path)

    # Default: treat as edge list
    return load_graph_from_edge_list(path)


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