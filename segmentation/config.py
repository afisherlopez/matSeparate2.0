from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class SamplingConfig:
   #dividing image into grid of patches

    type: str = "grid"  
    patch_size: int = 224  
    pad_mode: str = "reflect" 
    batch_size: int = 32  
    
    window_size: int = 96  
    window_area_pct: Optional[float] = None
    stride: Optional[int] = 48  
    min_patches: Optional[int] = None
    max_patches: Optional[int] = None


@dataclass
class UpsampleConfig:
    #bilinear upsampling
    mode: str = "bilinear"
    align_corners: bool = False
    renormalize: bool = True  


@dataclass
class CRFConfig:
    #dense CRF

    backend: str = "dense"  
    n_iterations: int = 7
    gaussian_sxy: float = 3.0
    bilateral_sxy: float = 60.0
    bilateral_srgb: float = 13.0
    gaussian_compat: float = 3.0
    bilateral_compat: float = 10.0
    taxonomy_aware_compat: bool = True
    # superpixel fallback only
    superpixel_method: str = "slic"  
    superpixel_n_segments: int = 400


@dataclass
class LevelConfig:
    """Which level of the taxonomy to segment at."""

    #default is leaf
    target: Union[str, int] = "leaf"
    shallow_leaf: str = "keep" 


@dataclass
class ObjectsConfig:
    """Label-map thresholding and connected-component object extraction."""

   #confidence threshold for classification
    bg_threshold: float = 0.0
    connectivity: int = 8  
    min_object_area: int = 64  
    morph_close: bool = False 
    morph_close_radius: int = 2


@dataclass
class OutputConfig:
    write_color_viz: bool = True
    write_instance_pngs: bool = False
    minc_crosswalk: Optional[str] = None 


@dataclass
class SegmentationConfig:

    run_dir: Optional[str] = None  
    checkpoint: Optional[str] = None  
    classifier: str = "legacy_hgnn"  
    device: str = "cpu"

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    upsample: UpsampleConfig = field(default_factory=UpsampleConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    level: LevelConfig = field(default_factory=LevelConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationConfig":
        data = dict(data or {})
        nested = {
            "sampling": SamplingConfig,
            "upsample": UpsampleConfig,
            "crf": CRFConfig,
            "level": LevelConfig,
            "objects": ObjectsConfig,
            "output": OutputConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in nested:
                sub_cls = nested[key]
                valid = {f.name for f in dataclasses.fields(sub_cls)}
                sub_kwargs = {k: v for k, v in (value or {}).items() if k in valid}
                kwargs[key] = sub_cls(**sub_kwargs)
            else:
                kwargs[key] = value
        valid_top = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_top}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SegmentationConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
