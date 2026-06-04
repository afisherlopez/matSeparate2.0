import torch

from patch_classifier.models import HGNNPatchClassifier, ResNetPatchClassifier
from patch_classifier.taxonomy import load_taxonomy


def test_resnet_shape():
    taxonomy = load_taxonomy("segmentation_pipeline/taxonomy/assets/matador-c1-taxonomy.json")
    model = ResNetPatchClassifier(num_classes=len(taxonomy.leaves), backbone="resnet18", pretrained=False)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, len(taxonomy.leaves))


def test_hgnn_shape():
    taxonomy = load_taxonomy("segmentation_pipeline/taxonomy/assets/matador-c1-taxonomy.json")
    model = HGNNPatchClassifier(taxonomy, backbone="resnet18", pretrained=False)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, taxonomy.num_nodes)

