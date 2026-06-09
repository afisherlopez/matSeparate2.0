"""Patch-classifier models.

Four classifiers, all over Matador-C1:
  - ResNetPatchClassifier:          local flat ResNet50, 37-way leaf head
  - GlobalResNetPatchClassifier:    local + context ResNet50, fused MLP head
  - HGNNPatchClassifier:            local image node + taxonomy graph, GAT
  - HGNNPatchClassifier(use_context=True): global-context HGNN

Fixed hyper params:
  - global ResNet head: hidden 1024, dropout 0.1
  - HGNN: image embedding 1024, 2-layer GAT, hidden 512, output 256,
          1 attention head, skip connections, dropout 0.1

The HGNN graph/objective design follows Beveridge et al. (2025).
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

from boilerplate.learning.taxonomy import Taxonomy

NUM_LEAVES = 37  # Matador-C1 leaf materials
IMAGE_EMBED_DIM = 1024


class ResNetPatchClassifier(nn.Module):
    """Local flat baseline: pretrained ResNet50 with a 37-way leaf head."""

    def __init__(self, num_classes: int = NUM_LEAVES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, image, context_image=None):
        return self.backbone(image)


class GlobalResNetPatchClassifier(nn.Module):
    """Global-context flat baseline.

    Two separate ResNet50 encoders (local crop + global context image). The
    pooled features are concatenated and passed through an MLP head:
        h = [f_local(x_local); f_context(x_context)]
    """

    def __init__(
        self,
        num_classes: int = NUM_LEAVES,
        pretrained: bool = True,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.context = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
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


class GraphBackbone(nn.Module):
    """Two-layer graph attention network with skip connections.

    First GAT layer: input_dim -> hidden_dim (num_heads, concatenated).
    Second GAT layer: hidden_dim*num_heads -> output_dim (1 head).
    Global mean pooling produces one graph-level embedding per image.
    """

    def __init__(
        self,
        input_dim: int = IMAGE_EMBED_DIM,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 1,
        skip_connection: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, concat=True,
            dropout=dropout, residual=skip_connection,
        )
        self.conv2 = GATConv(
            hidden_dim * num_heads, output_dim, heads=1, concat=False,
            dropout=dropout, residual=skip_connection,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class HGNNPatchClassifier(nn.Module):
    """Hierarchy-aware graph classifier.

    A ResNet50 image embedding is inserted as a graph node alongside one
    learned prototype node per taxonomy node. A 2-layer GAT propagates
    information between the image node and the material taxonomy, then a linear
    head produces logits over all taxonomy nodes.

    Edges: undirected taxonomy edges plus bidirectional image to taxonomy edges.
    With use_context=True the image node is built from concatenated local +
    context ResNet50 features.
    """

    def __init__(
        self,
        taxonomy: Taxonomy,
        pretrained: bool = True,
        image_embed_dim: int = IMAGE_EMBED_DIM,
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

        self.image_encoder = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.context_encoder = (
            timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            if use_context else None
        )
        enc_dim = self.image_encoder.num_features * (2 if use_context else 1)

        self.projection = nn.Linear(enc_dim, image_embed_dim)
        self.prototypes = nn.Embedding(self.num_nodes, image_embed_dim)
        self.gnn = GraphBackbone(
            input_dim=image_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            skip_connection=skip_connection,
            dropout=dropout,
        )
        self.classifier = nn.Linear(output_dim, self.num_nodes)

        # node 0 is the image node; taxonomy nodes are shifted by +1.
        taxonomy_edges = taxonomy.edge_index(undirected=True) + 1
        image_to_nodes = torch.stack(
            [
                torch.zeros(self.num_nodes, dtype=torch.long),
                torch.arange(1, self.num_nodes + 1, dtype=torch.long),
            ],
            dim=0,
        )
        nodes_to_image = image_to_nodes.flip(0)
        self.register_buffer(
            "edge_index",
            torch.cat([taxonomy_edges, image_to_nodes, nodes_to_image], dim=1),
        )

    def encode_image(self, image, context_image=None) -> torch.Tensor:
        feat = self.image_encoder(image)
        if self.use_context:
            if context_image is None:
                raise ValueError("HGNNPatchClassifier(use_context=True) needs context_image")
            feat = torch.cat([feat, self.context_encoder(context_image)], dim=1)
        return self.projection(feat)

    @torch.inference_mode()
    def init_prototypes(self, loader, device="cpu") -> None:
        """Initialize taxonomy-node prototypes from average CNN embeddings.

        Runs the (frozen) image encoder over the training set and sets each
        taxonomy node's prototype to the mean projected embedding of every
        example whose root-to-leaf path passes through that node. Nodes with no
        supporting examples keep their random initialization.
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
            node_target = batch["node_target"].to(device)  # (B, num_nodes) path multi-hot
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
