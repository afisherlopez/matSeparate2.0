"""Matador-C1 taxonomy wrapper.

The taxonomy is the 5-level Matador hierarchy from Beveridge et al. (2025):
phase -> state -> composition -> form -> material. We load it as a directed
tree (networkx) and expose the helpers the classifiers need:

  - node to index mapping over all taxonomy nodes
  - leaf list (the 37 Matador-C1 leaf materials)
  - multi-hot root-to-leaf path targets (for the HGNN objective)
  - per-level node index groups (for the level-wise loss term)
  - taxonomy edge index (for the HGNN graph)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import torch


@dataclass
class Taxonomy:
    graph: nx.DiGraph
    nodes: List[str]
    node_to_idx: Dict[str, int]
    root: str = "root"

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def leaves(self) -> List[str]:
        return [n for n in self.nodes if self.graph.out_degree(n) == 0]

    @property
    def leaf_indices(self) -> torch.Tensor:
        return torch.tensor(
            [self.node_to_idx[n] for n in self.leaves], dtype=torch.long
        )

    def path_to(self, label: str) -> List[str]:
        return nx.shortest_path(self.graph, self.root, label)

    def multihot_path(self, label: str) -> torch.Tensor:
        """Multi-hot vector marking every node on the root-to-leaf path."""
        target = torch.zeros(self.num_nodes, dtype=torch.float32)
        for node in self.path_to(label):
            target[self.node_to_idx[node]] = 1.0
        return target

    def edge_index(self, undirected: bool = True) -> torch.Tensor:
        edges = []
        for parent, child in self.graph.edges:
            edges.append((self.node_to_idx[parent], self.node_to_idx[child]))
            if undirected:
                edges.append((self.node_to_idx[child], self.node_to_idx[parent]))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def levels(self) -> List[torch.Tensor]:
        """Node-index groups at each taxonomy depth (root=0, leaves deepest)."""
        depth = nx.single_source_shortest_path_length(self.graph, self.root)
        out = []
        for d in sorted(set(depth.values())):
            nodes = [n for n, dd in depth.items() if dd == d]
            out.append(
                torch.tensor([self.node_to_idx[n] for n in nodes], dtype=torch.long)
            )
        return out


def load_taxonomy(path: str | Path, root: str = "root") -> Taxonomy:
    with open(path) as f:
        raw = json.load(f)
    graph = nx.tree_graph(raw, ident="name")
    nodes = sorted(graph.nodes())
    return Taxonomy(
        graph=graph,
        nodes=nodes,
        node_to_idx={n: i for i, n in enumerate(nodes)},
        root=root,
    )
