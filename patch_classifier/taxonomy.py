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
        return [node for node in self.nodes if self.graph.out_degree(node) == 0]

    @property
    def leaf_indices(self) -> torch.Tensor:
        return torch.tensor([self.node_to_idx[node] for node in self.leaves], dtype=torch.long)

    def path_to(self, label: str) -> List[str]:
        if label not in self.graph:
            raise ValueError(f"Label '{label}' is not in taxonomy")
        return nx.shortest_path(self.graph, self.root, label)

    def multihot_path(self, label: str) -> torch.Tensor:
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
        distances = nx.single_source_shortest_path_length(self.graph, self.root)
        out = []
        for level in sorted(set(distances.values())):
            nodes = [n for n, d in distances.items() if d == level]
            out.append(torch.tensor([self.node_to_idx[n] for n in nodes], dtype=torch.long))
        return out


def load_taxonomy(path: str | Path, root: str = "root") -> Taxonomy:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    graph = nx.tree_graph(raw, ident="name")
    if root not in graph:
        raise ValueError(f"Taxonomy root '{root}' not found in {path}")

    nodes = sorted(graph.nodes())
    return Taxonomy(
        graph=graph,
        nodes=nodes,
        node_to_idx={node: i for i, node in enumerate(nodes)},
        root=root,
    )

