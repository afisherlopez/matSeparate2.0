from typing import Dict

import networkx as nx
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_undirected

from taxonomy.tree import get_edge_index


class ImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        finetune: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.output_dim = output_dim

        backbone_model: nn.Module = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_channels,
        )
        classifier_input_dim = backbone_model.get_classifier().in_features
        self.cnn = backbone_model

        if hasattr(self.cnn, "fc"):
            delattr(self.cnn, "fc")
        elif hasattr(self.cnn, "head"):
            delattr(self.cnn, "head")

        if hasattr(self.cnn, "fc_norm"):
            delattr(self.cnn, "fc_norm")

        self.pooling = timm.layers.SelectAdaptivePool2d(
            pool_type="avg", flatten=nn.Flatten(start_dim=1, end_dim=-1),
        )
        self.classifier = nn.Linear(classifier_input_dim, output_dim)

        if finetune:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs) -> torch.Tensor:
        features = self.cnn.forward_features(images)
        features = self.pooling(features)
        return self.classifier(features)


class GraphBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 1,
        skip_connection: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList([
            GATConv(
                input_dim if i == 0 else hidden_dim * num_heads,
                hidden_dim if i < num_layers - 1 else output_dim,
                heads=num_heads if i < num_layers - 1 else 1,
                concat=i < num_layers - 1,
                dropout=dropout,
                residual=skip_connection,
            )
            for i in range(num_layers)
        ])

    def forward(self, node_features, edge_index, batch=None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            node_features = conv(node_features, edge_index)
            if i < len(self.convs) - 1:
                node_features = F.leaky_relu(node_features)
                node_features = self.dropout(node_features)

        if batch is not None:
            node_features = global_mean_pool(node_features, batch)

        return node_features.squeeze()


class HGNN(nn.Module):
    def __init__(
        self,
        graph: nx.DiGraph,
        dropout_prob: float = 0.0,
        cnn_kwargs: Dict = {},
        gnn_kwargs: Dict = {},
    ):
        super().__init__()
        self.num_nodes = graph.number_of_nodes()

        self.cnn = ImageEncoder(**cnn_kwargs)
        self.gnn = GraphBackbone(**gnn_kwargs)
        self.projection = nn.Linear(self.cnn.output_dim, self.gnn.input_dim)
        self.prototypes = nn.Embedding(self.num_nodes, self.gnn.input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.gnn.output_dim, self.num_nodes)

        self._init_graph(graph)

    def _init_graph(self, graph: nx.DiGraph):
        edge_index = to_undirected(
            torch.tensor(get_edge_index(graph), dtype=torch.long).t().contiguous()
        )
        context_to_labels = torch.stack([
            torch.zeros(graph.number_of_nodes(), dtype=torch.long),
            torch.arange(1, graph.number_of_nodes() + 1, dtype=torch.long),
        ], dim=0)
        edge_index_global_context = torch.cat([edge_index + 1, context_to_labels], dim=1)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_index_global_context", edge_index_global_context)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.projection(self.cnn(images))
        data_list = [
            Data(
                x=torch.cat([image_features[i].unsqueeze(0), self.prototypes.weight], dim=0),
                edge_index=self.edge_index_global_context,
            )
            for i in range(image_features.size(0))
        ]
        data_batch = Batch.from_data_list(data_list)
        node_embeddings = self.gnn(data_batch.x, data_batch.edge_index, batch=data_batch.batch)
        return self.classifier(self.dropout(node_embeddings))
