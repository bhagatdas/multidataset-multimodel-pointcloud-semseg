# config.py
import os
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration for point cloud semantic segmentation training."""

    # ----------------------------
    # Dataset
    # ----------------------------
    # Options: 'paris_lille_3d', 'toronto_3d', add more as needed
    dataset_name: str = "paris_lille_3d"

    paris_lille_path: str = "./data/preprocessed/Paris-Lille-3D"
    toronto_3d_path: str = "./data/preprocessed/Toronto-3D"

    use_preprocessed: bool = False

    @property
    def dataset_path(self) -> str:
        if self.dataset_name == "paris_lille_3d":
            return self.paris_lille_path
        elif self.dataset_name == "toronto_3d":
            return self.toronto_3d_path
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # Unified 5-class system (after preprocessing)
    unified_classes: List[str] = field(default_factory=lambda: [
        "Ground",
        "Building",
        "Vehicle",
        "Vegetation",
        "Unclassified",
    ])

    @property
    def class_names(self) -> List[str]:
        return self.unified_classes

    @property
    def num_classes(self) -> int:
        return len(self.unified_classes)

    @property
    def ignored_labels(self) -> List[int]:
        # customize if needed (e.g., [4] to ignore Unclassified)
        return []

    # ----------------------------
    # Model
    # ----------------------------
    # Options: 'pointnet', 'pointnet2', 'pointnet2_msg', 'pointcnn'
    model_type: str = "pointnet2"

    # ----------------------------
    # Point cloud sampling / blocks
    # ----------------------------
    num_points: int = 4096
    block_size: float = 1.0
    stride: float = 1.0
    load_blocks: bool = True

    @property
    def effective_block_size(self) -> float:
        # hook if you ever want to customize per-dataset/model
        return self.block_size

    @property
    def effective_stride(self) -> float:
        # hook if you ever want to customize per-dataset/model
        return self.stride

    # ----------------------------
    # Training
    # ----------------------------
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Hardware
    num_workers: int = 8
    use_cuda: bool = True
    multi_gpu: bool = False
    mixed_precision: bool = True

    # Optimization
    optimizer: str = "adam"       # placeholder for extensibility
    scheduler: str = "cosine"     # placeholder for extensibility
    loss_function: str = "combo"  # 'ce', 'focal', 'combo'
    focal_gamma: float = 2.0
    combo_alpha: float = 0.5
    feature_transform_reg: float = 0.001

    # Data augmentation
    use_augmentation: bool = True
    rotation_range: Tuple[float, float] = (0.0, 2.0 * 3.14159)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    jitter_std: float = 0.01
    class_balance_sampling: bool = True
    use_normization: bool = False  # (typo kept if used elsewhere)
    use_fps: bool = False

    # Class weights
    use_class_weights: bool = True

    # Checkpoint & logging
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_frequency: int = 5

    # Testing / eval
    test_batch_size: int = 8
    visualize_results: bool = True


config = Config()
