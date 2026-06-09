"""Project leaf-material probabilities to a chosen taxonomy level.

At the leaf level the projection is the identity. At an internal taxonomy level
(integer depth), each leaf probability is assigned to its ancestor at that depth
via a binary leaf-to-frontier aggregation matrix:

    P_frontier = P_leaf @ A,    A in {0, 1}^(L x K)

so the same dense probability map can produce segmentations at different
hierarchy depths without re-running the classifier.
"""

from dataclasses import dataclass
from typing import List, Union

import networkx as nx
import numpy as np


@dataclass
class FrontierCut:
    frontier_names: List[str]
    aggregation: np.ndarray  # (L, K), each leaf maps to exactly one frontier class
    level: Union[str, int]


def build_frontier(graph: nx.DiGraph, leaf_names: List[str],
                   level: Union[str, int] = "leaf", root: str = "root") -> FrontierCut:
    num_leaves = len(leaf_names)
    if level == "leaf":
        return FrontierCut(list(leaf_names), np.eye(num_leaves, dtype=np.float32), "leaf")

    if not isinstance(level, int) or level < 0:
        raise ValueError(f"level must be 'leaf' or a non-negative int depth, got {level!r}")

    leaf_to_rep: List[str] = []
    for leaf in leaf_names:
        path = nx.shortest_path(graph, root, leaf)
        # leaves shallower than the requested depth keep their own label
        leaf_to_rep.append(path[level] if (len(path) - 1) >= level else leaf)

    frontier_names = sorted(set(leaf_to_rep))
    rep_to_col = {name: k for k, name in enumerate(frontier_names)}
    agg = np.zeros((num_leaves, len(frontier_names)), dtype=np.float32)
    for i, rep in enumerate(leaf_to_rep):
        agg[i, rep_to_col[rep]] = 1.0
    return FrontierCut(frontier_names, agg, level)


def apply_frontier(p_leaf_dense: np.ndarray, cut: FrontierCut) -> np.ndarray:
    h, w, _ = p_leaf_dense.shape
    flat = p_leaf_dense.reshape(-1, p_leaf_dense.shape[-1])
    frontier = flat @ cut.aggregation
    return frontier.reshape(h, w, cut.aggregation.shape[1]).astype(np.float32)
