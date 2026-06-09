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
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.context = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.local.num_features
        self.head = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image, context_image=None):
        if context_image is None:
            raise ValueError("GlobalResNetPatchClassifier requires context_image")
        local_feat = self.local(image)
        context_feat = self.context(context_image)
        return self.head(torch.cat([local_feat, context_feat], dim=1))


class GATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 1,
        skip_connection: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_dim, hidden_dim, heads=num_heads, concat=True,
            dropout=dropout, residual=skip_connection,
        )
        self.conv2 = GATConv(
            hidden_dim * num_heads, out_dim, heads=1, concat=False,
            dropout=dropout, residual=skip_connection,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class HGNNPatchClassifier(nn.Module):
    def __init__(
        self,
        taxonomy: Taxonomy,
        backbone: str = "resnet50",
        pretrained: bool = True,
        image_embed_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 1,
        skip_connection: bool = True,
        dropout: float = 0.1,
        use_context: bool = False,
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.use_context = use_context
        self.num_nodes = taxonomy.num_nodes
        self.image_encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.context_encoder = (
            timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            if use_context else None
        )
        image_dim = self.image_encoder.num_features * (2 if use_context else 1)
        self.image_projector = nn.Linear(image_dim, image_embed_dim)
        self.prototypes = nn.Embedding(taxonomy.num_nodes, image_embed_dim)
        self.gnn = GATBlock(
            image_embed_dim, hidden_dim, output_dim,
            num_heads=num_heads, skip_connection=skip_connection, dropout=dropout,
        )
        self.classifier = nn.Linear(output_dim, taxonomy.num_nodes)

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

    def encode_image(self, image, context_image=None):
        image_feat = self.image_encoder(image)
        if self.use_context:
            if context_image is None:
                raise ValueError("HGNNPatchClassifier was built with use_context=True")
            context_feat = self.context_encoder(context_image)
            image_feat = torch.cat([image_feat, context_feat], dim=1)
        return self.image_projector(image_feat)

    @torch.inference_mode()
    def init_prototypes(self, loader, device="cpu"):
        """Initialize taxonomy-node prototypes from average CNN embeddings.

        Each prototype is set to the mean projected image embedding over all
        training examples whose root-to-leaf path passes through that node.
        """
        self.eval()
        dim = self.prototypes.embedding_dim
        sums = torch.zeros(self.num_nodes, dim, device=device)
        counts = torch.zeros(self.num_nodes, device=device)
        for batch in loader:
            image = batch["image"].to(device)
            context = batch.get("context_image")
            context = context.to(device) if context is not None else None
            feat = self.encode_image(image, context)
            node_target = batch["node_target"].to(device)
            sums += node_target.t() @ feat
            counts += node_target.sum(dim=0)
        seen = counts > 0
        self.prototypes.weight.data[seen] = sums[seen] / counts[seen].unsqueeze(1)

    def forward(self, image, context_image=None):
        image_feat = self.encode_image(image, context_image)
        data_list = []
        for i in range(image_feat.shape[0]):
            x = torch.cat([image_feat[i : i + 1], self.prototypes.weight], dim=0)
            data_list.append(Data(x=x, edge_index=self.edge_index))
        batch = Batch.from_data_list(data_list)
        graph_feat = self.gnn(batch.x, batch.edge_index, batch.batch)
        return self.classifier(graph_feat)

