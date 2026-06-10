<<<<<<< HEAD
from __future__ import annotations

=======
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
from dataclasses import dataclass
from typing import List, Union

import networkx as nx
import numpy as np


@dataclass
class FrontierCut:
<<<<<<< HEAD

    frontier_names: List[str] 
    aggregation: np.ndarray  
    leaf_names: List[str]  
=======
    frontier_names: List[str]
    aggregation: np.ndarray  # (L, K), each leaf maps to exactly one frontier class
    leaf_names: List[str]
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
    level: Union[str, int]


def build_frontier(
    graph: nx.DiGraph,
    leaf_names: List[str],
    level: Union[str, int] = "leaf",
    shallow_leaf: str = "keep",
    root: str = "root",
) -> FrontierCut:
    num_leaves = len(leaf_names)

    if level == "leaf":
        agg = np.eye(num_leaves, dtype=np.float32)
        return FrontierCut(list(leaf_names), agg, list(leaf_names), "leaf")

    if not isinstance(level, int):
        raise ValueError(f"level must be 'leaf' or an int depth, got {level!r}")
    if level < 0:
        raise ValueError("integer level (depth) must be >= 0")

<<<<<<< HEAD
    depths = _node_depths(graph, root)

    #map each leaf to representative node at requested depth
=======
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
    leaf_to_rep: List[str] = []
    for leaf in leaf_names:
        if leaf not in graph:
            raise ValueError(f"leaf '{leaf}' not found in taxonomy graph")
<<<<<<< HEAD
        path = nx.shortest_path(graph, root, leaf)  
=======
        path = nx.shortest_path(graph, root, leaf)
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
        leaf_depth = len(path) - 1
        if leaf_depth >= level:
            rep = path[level]
        else:
<<<<<<< HEAD
            #leaf is shallower than requested depth
=======
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
            if shallow_leaf == "keep":
                rep = leaf
            elif shallow_leaf == "parent":
                rep = path[-2] if len(path) >= 2 else leaf
            else:
                raise ValueError(f"unknown shallow_leaf mode: {shallow_leaf}")
        leaf_to_rep.append(rep)

    frontier_names = sorted(set(leaf_to_rep))
    rep_to_col = {name: k for k, name in enumerate(frontier_names)}

    agg = np.zeros((num_leaves, len(frontier_names)), dtype=np.float32)
    for i, rep in enumerate(leaf_to_rep):
        agg[i, rep_to_col[rep]] = 1.0

    return FrontierCut(frontier_names, agg, list(leaf_names), level)


def apply_frontier(p_leaf_dense: np.ndarray, cut: FrontierCut) -> np.ndarray:
    if p_leaf_dense.shape[-1] != cut.aggregation.shape[0]:
        raise ValueError(
            f"leaf dim {p_leaf_dense.shape[-1]} != aggregation rows "
            f"{cut.aggregation.shape[0]}"
        )
    h, w, _ = p_leaf_dense.shape
    flat = p_leaf_dense.reshape(-1, p_leaf_dense.shape[-1])
<<<<<<< HEAD
    frontier = flat @ cut.aggregation 
=======
    frontier = flat @ cut.aggregation
>>>>>>> c01bc9a0d0c1709b7dca3bfcc70c1cedcb4ee065
    return frontier.reshape(h, w, cut.aggregation.shape[1]).astype(np.float32)
