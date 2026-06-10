import json
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


def get_taxonomy(
    source: Union[str, Path, Dict] = Path(__file__)
    .resolve()
    .parent.joinpath("assets", "taxonomy-tree.json")
) -> nx.DiGraph:
    if isinstance(source, dict):
        tree_data = source
    elif isinstance(source, (str, Path)):
        with open(source, "r") as f:
            tree_data = json.load(f)
    else:
        raise TypeError("source must be a string, Path-like object, or dictionary.")

    return nx.tree_graph(tree_data, ident="name")


def modify_graph(
    graph: nx.DiGraph, node_map: Dict[str, Tuple[str, List[Tuple[str, Dict[str, Any]]]]]
):
    for original_node, (mode, new_nodes) in node_map.items():
        if original_node not in graph:
            raise ValueError(f"Node '{original_node}' does not exist in graph.")
        for new_node, new_attrs in new_nodes:
            if mode == "rename":
                nx.relabel_nodes(graph, {original_node: new_node})
            elif mode == "child":
                graph.add_node(new_node, **new_attrs)
                graph.add_edge(original_node, new_node)
            elif mode == "duplicate":
                graph.add_node(new_node, **{**graph.nodes[original_node], **new_attrs})
                for predecessor in graph.predecessors(original_node):
                    graph.add_edge(
                        predecessor, new_node, **graph.edges[predecessor, original_node]
                    )
                for successor in graph.successors(original_node):
                    graph.add_edge(
                        new_node, successor, **graph.edges[original_node, successor]
                    )
            else:
                raise TypeError(
                    f"Invalid mode: {mode}. Must be 'rename', 'child', or 'duplicate'."
                )


def shuffle_graph(graph: nx.Graph, mode: str = "both") -> nx.Graph:
    if mode not in ["nodes", "edges", "both"]:
        raise Exception(f"Invalid mode: {mode}")

    shuffled_graph = graph.copy()

    if mode == "nodes" or "both":
        nodes = list(graph.nodes)
        permuted_nodes = nodes[:]
        np.random.shuffle(permuted_nodes)
        node_mapping = {old: new for old, new in zip(nodes, permuted_nodes)}
        permuted_graph = nx.relabel_nodes(graph, node_mapping)
    else:
        permuted_graph = graph.copy()

    if mode == "edges" or "both":
        shuffled_edges = list(permuted_graph.edges)
        np.random.shuffle(shuffled_edges)
        shuffled_graph.add_nodes_from(permuted_graph.nodes)
        shuffled_graph.add_edges_from(shuffled_edges)
    else:
        shuffled_graph = permuted_graph.copy()

    return shuffled_graph


def get_edge_index(graph: nx.DiGraph) -> np.ndarray:
    edge_index = np.array(taxa_to_indices(graph.edges, graph))
    return edge_index


def get_hierarchy_levels(graph: nx.DiGraph, root_name: Any = "root") -> Dict:
    result = {}
    for k, v in nx.single_source_shortest_path_length(graph, root_name).items():
        result[v] = result.get(v, []) + [k]
    return result


def get_hierarchy_mask(
    graph: nx.DiGraph, self_referential: bool = True, mode="descendants"
) -> np.ndarray:
    nodes = list(graph)
    n = len(nodes)
    hierarchy_mask = np.zeros((n, n), dtype=int)
    node_index = {node: i for i, node in enumerate(nodes)}

    if mode == "descendants":
        get_children = partial(nx.descendants, graph)
    elif mode == "successors":
        get_children = graph.successors
    else:
        raise ValueError(f"Invalid mode: {mode}.")

    for parent in nodes:
        children = set(get_children(parent))
        if self_referential:
            children = children | {parent}
        for child in children:
            hierarchy_mask[node_index[parent], node_index[child]] = 1

    return hierarchy_mask


def onehot_to_taxa(
    onehot: np.ndarray, graph: nx.DiGraph, full_path: bool = True, **kwargs
) -> np.ndarray:
    nodes = np.array(list(graph))
    maxlevel = kwargs.get(
        "maxlevel",
        max(nx.single_source_shortest_path_length(graph, "root").values()) + 1,
    )

    taxa = []
    for oh in onehot:
        path = nodes[oh > 0]
        if nx.is_path(graph, path):
            taxa.append(path[:maxlevel] if full_path else path[-1])
        else:
            taxa.append("")

    return np.array(taxa)


def taxa_to_onehot(
    taxa: np.ndarray, graph: nx.DiGraph, full_path: bool = True, **kwargs
) -> np.ndarray:
    nodes = list(graph)
    minlevel = kwargs.get("minlevel", 0)
    maxlevel = kwargs.get(
        "maxlevel",
        max(nx.single_source_shortest_path_length(graph, "root").values()) + 1,
    )

    def _get_one_hot(s):
        index = nodes.index(s)
        one_hot = np.zeros(len(nodes))
        one_hot[index] = 1
        return one_hot

    if full_path:
        taxa = np.array(
            [
                (sp := nx.shortest_path(graph, "root", t)[minlevel:maxlevel])
                + [""] * (maxlevel - len(sp))
                for t in taxa
            ]
        )

    flat_array = taxa.flatten()
    one_hot_vectors = np.array(
        [_get_one_hot(s) if s else np.zeros(len(nodes)) for s in flat_array]
    )
    result = one_hot_vectors.reshape(taxa.shape + (len(nodes),))
    if full_path:
        result = result.sum(axis=1)

    return result


def taxa_to_indices(taxa: np.ndarray, graph: nx.DiGraph) -> np.ndarray:
    nodes = list(graph)
    name_to_idx = np.vectorize(lambda s: nodes.index(s))
    return name_to_idx(taxa)


def indices_to_taxa(indices: np.ndarray, graph: nx.DiGraph) -> np.ndarray:
    nodes = np.array(list(graph))
    return nodes[indices]


def aggregate_graph_attributes(
    graph: nx.DiGraph,
    start_node: str = "root",
    node_agg_func: Callable = sum,
    edge_agg_func: Callable = sum,
) -> nx.DiGraph:
    descendants = nx.descendants(graph, start_node) | {start_node}
    subgraph = graph.subgraph(descendants).copy()

    for node in list(nx.topological_sort(subgraph))[::-1]:
        for child in subgraph.successors(node):
            for attr, value in subgraph.nodes[child].items():
                if attr in subgraph.nodes[node]:
                    subgraph.nodes[node][attr] = node_agg_func(
                        subgraph.nodes[node][attr], value
                    )
                else:
                    subgraph.nodes[node][attr] = value

            for attr, value in subgraph.get_edge_data(node, child, default={}).items():
                if attr in subgraph.nodes[node]:
                    subgraph.nodes[node][attr] = edge_agg_func(
                        subgraph.nodes[node][attr], value
                    )
                else:
                    subgraph.nodes[node][attr] = value

    return subgraph


def graph_to_nested_list(graph: nx.DiGraph, root=None):
    if root is None:
        candidates = [n for n in graph.nodes if graph.in_degree(n) == 0]
        if not candidates:
            raise ValueError("The graph must have at least one root node.")
        root = candidates[0]

    def build_tuple(node):
        children = list(graph.successors(node))
        if not children:
            return [node.capitalize()]
        return [node.capitalize()] + [build_tuple(child) for child in children]

    return build_tuple(root)


def all_paths_subgraph(graph: nx.DiGraph, root: str, targets: List[str]) -> nx.DiGraph:
    edges = set()
    for target in targets:
        for path in nx.all_simple_paths(graph, source=root, target=target):
            edges.update(zip(path[:-1], path[1:]))
    return graph.edge_subgraph(edges).copy()


def get_minimal_branching_subgraph(
    graph: nx.DiGraph, target_nodes: List[str]
) -> nx.DiGraph:
    frozen_subG = nx.algorithms.approximation.steiner_tree(
        graph.to_undirected(), target_nodes
    )

    subG = nx.Graph(frozen_subG) if not graph.is_directed() else nx.DiGraph(frozen_subG)

    prunable_nodes = {
        node
        for node in subG.nodes
        if subG.degree(node) == 2 and node not in target_nodes
    }

    while prunable_nodes:
        node = prunable_nodes.pop()
        neighbors = list(subG.neighbors(node))
        if len(neighbors) == 2:
            subG.add_edge(neighbors[0], neighbors[1])
        subG.remove_node(node)
        prunable_nodes = {
            n for n in subG.nodes if subG.degree(n) == 2 and n not in target_nodes
        }

    return subG


def level_complete_tree(graph: nx.Graph, root: Any, directed: bool = False) -> nx.Graph:
    if directed:
        new_graph = nx.DiGraph()
    else:
        new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes(data=True))
    new_graph.add_edges_from(graph.edges(data=True))

    depth: Dict[Any, int] = nx.single_source_shortest_path_length(graph, root)
    levels: Dict[int, List[Any]] = {}
    for node, d in depth.items():
        levels.setdefault(d, []).append(node)

    for nodes_at_level in levels.values():
        for u, v in combinations(nodes_at_level, 2):
            new_graph.add_edge(u, v)
            if directed:
                new_graph.add_edge(v, u)

    return new_graph


def write_network_text_with_color(
    graph: nx.Graph,
    colored_nodes: Iterable = [],
    path: Optional[str] = None,
    highlight_color: str = "\033[34m",
    reset_color: str = "\033[0m",
    **kwargs,
) -> Union[str, None]:
    colored_nodes_str = set(str(n) for n in colored_nodes)
    lines = list(nx.generate_network_text(graph, **kwargs))
    output_lines = []

    for line in lines:
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            prefix, last_token = parts
            if last_token in colored_nodes_str:
                output_lines.append(f"{prefix} {highlight_color}{last_token}{reset_color}")
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)

    result = "\n".join(output_lines)

    if path:
        with open(path, "w") as f:
            f.write(result)
        return None
    return result
