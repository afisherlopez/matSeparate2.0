from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from patch_classifier.models import (
    GlobalResNetPatchClassifier,
    HGNNPatchClassifier,
    ResNetPatchClassifier,
)
from patch_classifier.taxonomy import Taxonomy, load_taxonomy
from patch_classifier.train import make_model


class PatchClassifierInference:
    """Small inference wrapper for checkpoints trained by patch_classifier.train."""

    def __init__(self, model, taxonomy: Taxonomy, config: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.taxonomy = taxonomy
        self.config = config
        self.device = device
        self.leaf_indices = taxonomy.leaf_indices.to(device)
        self.leaf_names: List[str] = list(taxonomy.leaves)
        image_size = config.get("training", {}).get("image_size", 224)
        self.transform = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.CenterCrop(image_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str = "auto"):
        checkpoint = Path(checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu")
        config = ckpt["config"]
        taxonomy = load_taxonomy(config["taxonomy"], root=config.get("root", "root"))
        model = make_model(config, taxonomy)
        state = ckpt.get("model", ckpt.get("model_state_dict"))
        if state is None:
            raise ValueError(f"No model state found in {checkpoint}")
        model.load_state_dict(state)
        return cls(model, taxonomy, config, _resolve_device(device))

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: str = "auto"):
        run_dir = Path(run_dir)
        for name in ("best.pt", "checkpoint_best.pt"):
            path = run_dir / name
            if path.exists():
                return cls.from_checkpoint(path, device=device)
        raise FileNotFoundError(f"No best.pt or checkpoint_best.pt found in {run_dir}")

    @torch.inference_mode()
    def infer_batch(self, images, return_probs: bool = True, decode_path: bool = False):
        batch = torch.stack([self._prepare_image(im) for im in images]).to(self.device)
        logits = self.model(batch)

        if logits.shape[1] == self.taxonomy.num_nodes:
            leaf_logits = logits[:, self.leaf_indices]
            leaf_probs = torch.softmax(leaf_logits, dim=1)
            leaf_pos = leaf_probs.argmax(dim=1)
            leaf_node_idx = self.leaf_indices[leaf_pos].cpu().tolist()
            leaf_labels = [self.taxonomy.nodes[i] for i in leaf_node_idx]
        else:
            leaf_probs = torch.softmax(logits, dim=1)
            leaf_pos = leaf_probs.argmax(dim=1).cpu().tolist()
            leaf_labels = [self.leaf_names[i] for i in leaf_pos]

        probs_np = leaf_probs.cpu().numpy()
        out = []
        for i, label in enumerate(leaf_labels):
            item = {"leaf_label": label}
            if return_probs:
                item["leaf_probs"] = probs_np[i]
            if decode_path:
                item["path_nodes"] = self.taxonomy.path_to(label)
            out.append(item)
        return out

    def _prepare_image(self, image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return self.transform(tensor)


def _resolve_device(device: str):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

