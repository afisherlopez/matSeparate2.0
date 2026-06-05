import json
from pathlib import Path
from typing import Any, Dict, Union

import networkx as nx
import numpy as np


def get_taxonomy(
    source: Union[str, Path, Dict] = Path(__file__).resolve().parent.joinpath(
        "assets", "taxonomy-tree.json"
    )
) -> nx.DiGraph:
    if isinstance(source, dict):
        tree_data = source
    elif isinstance(source, (str, Path)):
        with open(source, "r") as f:
            tree_data = json.load(f)
    else:
        raise TypeError("source must be a string, Path-like object, or dictionary.")

    return nx.tree_graph(tree_data, ident="name")


def get_edge_index(graph: nx.DiGraph) -> np.ndarray:
    nodes = list(graph)
    return np.array([(nodes.index(u), nodes.index(v)) for u, v in graph.edges])


def get_hierarchy_levels(graph: nx.DiGraph, root_name: Any = "root") -> Dict:
    result = {}
    for k, v in nx.single_source_shortest_path_length(graph, root_name).items():
        result[v] = result.get(v, []) + [k]
    return result
