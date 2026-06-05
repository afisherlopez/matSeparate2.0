import torch
import torch.nn.functional as F


def hgnn_loss(node_logits, node_targets, levels):
    """Small version of the Beveridge-style path + per-level objective."""
    bce = F.binary_cross_entropy_with_logits(node_logits, node_targets)

    level_losses = []
    for level in levels:
        level = level.to(node_logits.device)
        targets = node_targets[:, level]
        participates = targets.sum(dim=1) > 0
        if participates.any():
            level_logits = node_logits[participates][:, level]
            level_target = targets[participates].argmax(dim=1)
            level_losses.append(F.cross_entropy(level_logits, level_target))

    if not level_losses:
        return bce
    return torch.maximum(bce, torch.stack(level_losses).mean())


def leaf_accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def hgnn_leaf_accuracy(node_logits, node_targets, leaf_indices):
    leaf_indices = leaf_indices.to(node_logits.device)
    leaf_targets = node_targets[:, leaf_indices].argmax(dim=1)
    leaf_preds = node_logits[:, leaf_indices].argmax(dim=1)
    return (leaf_preds == leaf_targets).float().mean().item()

