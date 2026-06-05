import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

from classifier.taxonomy import Taxonomy


class ResNetPatchClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, image, context_image=None):
        return self.backbone(image)


class GlobalResNetPatchClassifier(nn.Module):
    """Two ResNet encoders. One sees the local patch, the other sees wider context."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.local = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.context = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.local.num_features
        self.head = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image, context_image=None):
        if context_image is None:
            raise ValueError("GlobalResNetPatchClassifier requires context_image")
        local_feat = self.local(image)
        context_feat = self.context(context_image)
        return self.head(torch.cat([local_feat, context_feat], dim=1))


class GATBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=True, dropout=dropout)
        self.conv2 = GATConv(2 * hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class HGNNPatchClassifier(nn.Module):
    """End-to-end image encoder + taxonomy graph classifier.

    A learned node embedding is kept for every taxonomy node. For each image, we add
    one image node connected to every taxonomy node, run a small GAT, then predict
    logits for all taxonomy nodes.
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_context: bool = False,
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.use_context = use_context
        self.image_encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.context_encoder = (
            timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            if use_context else None
        )
        image_dim = self.image_encoder.num_features * (2 if use_context else 1)
        self.image_projector = nn.Linear(image_dim, embed_dim)
        self.node_embeddings = nn.Embedding(taxonomy.num_nodes, embed_dim)
        self.gnn = GATBlock(embed_dim, hidden_dim, embed_dim, dropout=dropout)
        self.classifier = nn.Linear(embed_dim, taxonomy.num_nodes)

        graph_edges = taxonomy.edge_index(undirected=True) + 1
        image_edges = torch.stack(
            [
                torch.zeros(taxonomy.num_nodes, dtype=torch.long),
                torch.arange(1, taxonomy.num_nodes + 1, dtype=torch.long),
            ],
            dim=0,
        )
        reverse_image_edges = image_edges.flip(0)
        self.register_buffer("edge_index", torch.cat([graph_edges, image_edges, reverse_image_edges], dim=1))

    def forward(self, image, context_image=None):
        image_feat = self.image_encoder(image)
        if self.use_context:
            if context_image is None:
                raise ValueError("HGNNPatchClassifier was built with use_context=True")
            context_feat = self.context_encoder(context_image)
            image_feat = torch.cat([image_feat, context_feat], dim=1)

        image_feat = self.image_projector(image_feat)
        data_list = []
        for i in range(image_feat.shape[0]):
            x = torch.cat([image_feat[i : i + 1], self.node_embeddings.weight], dim=0)
            data_list.append(Data(x=x, edge_index=self.edge_index))
        batch = Batch.from_data_list(data_list)
        graph_feat = self.gnn(batch.x, batch.edge_index, batch.batch)
        return self.classifier(graph_feat)

