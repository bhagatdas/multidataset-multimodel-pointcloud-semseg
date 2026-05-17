import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from plyfile import PlyData
from pathlib import Path
import numpy as np
from tqdm import tqdm  # NEW

from utils.data_utils import PointCloudProcessor, DataAugmentation


class PreprocessedDataset(Dataset):
    """
    Dataset for preprocessed .ply segments.

    - Uses moving-window blocks (stride can be < block_size for overlap).
    - For train, can use multiple block-generation modes (grid, random shapes, multi-scale).
    - Everything is built and augmented on the fly.
    - Returns (points, labels, mean, std) per block.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        num_points: int = 8192,
        block_size: float = 50.0,
        stride: float = 25.0,
        num_classes: int = 5,
        use_augmentation: bool = True,
        augmentation_args: dict = None,
        min_points_per_block: int = 100,
        use_normization: bool = False,
        use_fps: bool = True,
        use_multi_view_blocks: bool = False,
        random_shape_config: dict = None,
        multi_scale_block_sizes: tuple = (1.0, 100.0),
    
        load: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.num_points = num_points
        self.block_size = float(block_size)
        self.stride = float(stride)
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation
        self.min_points_per_block = min_points_per_block
        self.use_normization = use_normization
        self.use_fps = use_fps

        self.use_multi_view_blocks = use_multi_view_blocks
        self.random_shape_config = random_shape_config or dict(
            min_width=10.0,
            max_width=80.0,
            min_height=10.0,
            max_height=80.0,
            num_random_windows=50,
        )
        self.multi_scale_block_sizes = multi_scale_block_sizes
        self.load = load

        if augmentation_args is None:
            augmentation_args = dict(
                use_rotation=True,
                use_scaling=True,
                use_jitter=True,
                use_flip=True,
                use_dropout=False,
                rotation_3d=False,
                scale_range=(0.8, 1.2),
                jitter_sigma=0.01,
                jitter_clip=0.05,
                dropout_ratio=0.2,
            )
        self.augmentor = DataAugmentation(**augmentation_args)

        # ------------------------------------------------------------------
        # Discover .ply files for this split
        # ------------------------------------------------------------------
        split_dir = self.dataset_path / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # NEW: cache file for this split
        cache_file = split_dir / f"{split}_blocks_cache.pt"

        # ------------------------------------------------------------------
        # Try to load cached blocks if load=True
        # ------------------------------------------------------------------
        if self.load and cache_file.exists():
            print(f"[INFO] Loading cached blocks for split='{split}' from {cache_file}")
            cache = torch.load(cache_file, map_location="cpu")
            self.data_blocks = cache["data_blocks"]
            self.num_segments = cache["num_segments"]
            self.raw_point_count = cache["raw_point_count"]

            print(f"Total blocks: {len(self.data_blocks)} (from cache)")
            self._calculate_class_weights()
            print(f"Class weights: {self.class_weights}")
            # self._print_split_summary()
            return

        # ------------------------------------------------------------------
        # Normal path: build blocks from PLYs
        # ------------------------------------------------------------------
        segment_files = sorted(split_dir.glob("*.ply"))
        if len(segment_files) == 0:
            raise ValueError(f"No PLY segment files found in {split_dir}")

        self.num_segments = len(segment_files)
        self.raw_point_count = 0  # total points in all PLYs for this split

        print(f"Loading {split} split from {split_dir}")
        print(f"Found {self.num_segments} PLY segment files")

        # Warn if stride is suspicious
        if self.stride > self.block_size:
            print(
                f"[WARN] For split='{self.split}', stride ({self.stride}) > "
                f"block_size ({self.block_size}). This may lead to very low "
                f"coverage of points in blocks."
            )

        # ------------------------------------------------------------------
        # Build blocks using different strategies
        # ------------------------------------------------------------------
        self.data_blocks = []

        for seg_file in tqdm(segment_files, desc=f"Building blocks [{self.split}]", ncols=80):
            points, labels = self._read_ply(seg_file)
            self.raw_point_count += points.shape[0]

            if self.split == "train" and self.use_multi_view_blocks:
                # Mode 1: equal-size grid blocks using (block_size, stride)
                blocks1 = self._build_equal_grid_blocks(points, labels)

                # Mode 2: irregular shapes and sizes, overlapping, independent of block_size
                blocks2 = self._build_random_shape_blocks(points, labels)

                # Mode 3: multi-scale blocks with random block sizes
                blocks3 = self._build_multi_scale_blocks(points, labels)

                self.data_blocks.extend(blocks1)
                self.data_blocks.extend(blocks2)
                self.data_blocks.extend(blocks3)
            else:
                # For val/test (and optionally train if multi-view disabled):
                blocks = self._build_equal_grid_blocks(points, labels)
                self.data_blocks.extend(blocks)

        print(f"Total blocks: {len(self.data_blocks)}")

        # Class weights and split summary
        self._calculate_class_weights()
        print(f"Class weights: {self.class_weights}")
        # self._print_split_summary()

        # ------------------------------------------------------------------
        # NEW: save cached blocks if load=True
        # ------------------------------------------------------------------
        if self.load:
            cache = {
                "data_blocks": self.data_blocks,
                "num_segments": self.num_segments,
                "raw_point_count": self.raw_point_count,
            }
            torch.save(cache, cache_file)
            print(f"[INFO] Cached blocks saved to {cache_file}")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _read_ply(self, ply_path: Path):
        plydata = PlyData.read(str(ply_path))
        v = plydata["vertex"].data

        # XYZ only; extend here if you have more features
        features = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
        labels = np.array(v["label"]).astype(np.int64)

        points = torch.from_numpy(features)  # [N, 3]
        labels = torch.from_numpy(labels)    # [N]
        return points, labels

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------
    def _build_equal_grid_blocks(self, points, labels):
        """
        Mode 1: Regular grid blocks in XY, based on (block_size, stride).
        """
        blocks = []

        min_coords = torch.min(points[:, 0:2], dim=0)[0]
        max_coords = torch.max(points[:, 0:2], dim=0)[0]

        x_range = max_coords[0] - min_coords[0]
        y_range = max_coords[1] - min_coords[1]

        x_steps = max(int(torch.ceil(x_range / self.stride).item()), 1)
        y_steps = max(int(torch.ceil(y_range / self.stride).item()), 1)

        for i in tqdm(range(x_steps), desc="Grid-X", leave=False, ncols=80):
            for j in range(y_steps):
                x_min = min_coords[0] + i * self.stride
                y_min = min_coords[1] + j * self.stride
                x_max = x_min + self.block_size
                y_max = y_min + self.block_size

                mask = (
                    (points[:, 0] >= x_min)
                    & (points[:, 0] < x_max)
                    & (points[:, 1] >= y_min)
                    & (points[:, 1] < y_max)
                )
                block_points = points[mask]
                block_labels = labels[mask]

                if block_points.shape[0] < self.min_points_per_block:
                    continue

                # Center the block
                block_points = block_points.clone()
                block_points[:, :3] -= torch.mean(block_points[:, :3], dim=0)

                blocks.append((block_points, block_labels))

        return blocks

    def _build_random_shape_blocks(self, points, labels):
        """
        Mode 2: Random XY rectangles with random width/height,
        not tied to block_size, overlapping windows.
        """
        cfg = self.random_shape_config
        blocks = []

        min_xy = torch.min(points[:, :2], dim=0)[0]
        max_xy = torch.max(points[:, :2], dim=0)[0]

        for _ in tqdm(range(cfg["num_random_windows"]), desc="RandomShapes", leave=False, ncols=80):
            # Random center in XY range
            center_x = float(torch.empty(1).uniform_(min_xy[0], max_xy[0]))
            center_y = float(torch.empty(1).uniform_(min_xy[1], max_xy[1]))

            # Random width/height
            width = float(torch.empty(1).uniform_(cfg["min_width"], cfg["max_width"]))
            height = float(torch.empty(1).uniform_(cfg["min_height"], cfg["max_height"]))

            x_min = center_x - width / 2.0
            x_max = center_x + width / 2.0
            y_min = center_y - height / 2.0
            y_max = center_y + height / 2.0

            mask = (
                (points[:, 0] >= x_min)
                & (points[:, 0] < x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] < y_max)
            )
            block_points = points[mask]
            block_labels = labels[mask]

            if block_points.shape[0] < self.min_points_per_block:
                continue

            block_points = block_points.clone()
            block_points[:, :3] -= torch.mean(block_points[:, :3], dim=0)

            blocks.append((block_points, block_labels))

        return blocks

    def _build_multi_scale_blocks(self, points, labels):
        """
        Mode 3: Multi-scale blocks with random square sizes
        in [min_size, max_size], overlapping.
        """
        blocks = []

        min_size, max_size = self.multi_scale_block_sizes
        num_windows = 50  # can be tuned

        min_xy = torch.min(points[:, :2], dim=0)[0]
        max_xy = torch.max(points[:, :2], dim=0)[0]

        for _ in tqdm(range(num_windows), desc="MultiScale", leave=False, ncols=80):
            # Random block size
            size = float(torch.empty(1).uniform_(min_size, max_size))

            # Random center
            center_x = float(torch.empty(1).uniform_(min_xy[0], max_xy[0]))
            center_y = float(torch.empty(1).uniform_(min_xy[1], max_xy[1]))

            x_min = center_x - size / 2.0
            x_max = center_x + size / 2.0
            y_min = center_y - size / 2.0
            y_max = center_y + size / 2.0

            mask = (
                (points[:, 0] >= x_min)
                & (points[:, 0] < x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] < y_max)
            )
            block_points = points[mask]
            block_labels = labels[mask]

            if block_points.shape[0] < self.min_points_per_block:
                continue

            block_points = block_points.clone()
            block_points[:, :3] -= torch.mean(block_points[:, :3], dim=0)

            blocks.append((block_points, block_labels))

        return blocks

    # ------------------------------------------------------------------
    # Class weights & distributions
    # ------------------------------------------------------------------
    def _compute_class_histogram(self):
        """
        Count number of points per class across ALL blocks
        (i.e., after block generation, before num_points sampling).
        """
        label_counts = torch.zeros(self.num_classes, dtype=torch.long)

        for _, lbls in self.data_blocks:
            lbls = lbls.view(-1)
            valid = (lbls >= 0) & (lbls < self.num_classes)
            lbls = lbls[valid]

            unique, counts = torch.unique(lbls, return_counts=True)
            label_counts[unique] += counts

        return label_counts

    def _calculate_class_weights(self):
        """
        Compute per-class weights from frequencies (soft reweighting).
        """
        label_counts = self._compute_class_histogram().to(torch.float32)

        total_points = label_counts.sum()
        if total_points <= 0:
            # Avoid division by zero; fall back to uniform
            self.class_weights = torch.ones(self.num_classes, dtype=torch.float32)
            return

        freqs = label_counts / (total_points + 1e-6)
        max_freq = freqs.max()
        beta = 1.0 / 3.0  # soften imbalance
        class_weights = (max_freq / (freqs + 1e-6)) ** beta
        self.class_weights = class_weights

    def _print_split_summary(self):
        """
        Pretty print: num segments, raw points, points in blocks,
        coverage, per-class counts & percentages.
        """
        label_counts = self._compute_class_histogram()
        points_in_blocks = label_counts.sum().item()
        raw_points = int(self.raw_point_count)

        coverage = (
            100.0 * points_in_blocks / raw_points if raw_points > 0 else 0.0
        )

        print(f"\n[{self.split}] Split Summary")
        print(f"  PLY segments         : {self.num_segments}")
        print(f"  Blocks               : {len(self.data_blocks)}")
        print(f"  Raw points (all PLYs): {raw_points}")
        print(f"  Points in blocks     : {points_in_blocks}")
        print(f"  Coverage             : {coverage:.2f}%")
        print(f"  Class counts :")
        for c in range(self.num_classes):
            count = label_counts[c].item()
            pct = (
                100.0 * count / points_in_blocks if points_in_blocks > 0 else 0.0
            )
            print(f"    Class {c}: {count} pts ({pct:.2f}%)")
        print()

    # ------------------------------------------------------------------
    # Extra spatial augmentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _multi_scale_random_crop(points, labels, block_size, min_points_per_block):
        """
        Random multi-scale crop around a random center point in XY.
        """
        if points.shape[0] < min_points_per_block:
            return points, labels

        center_idx = torch.randint(points.shape[0], (1,)).item()
        center_xy = points[center_idx, :2]

        # Random scale between 0.6x and 1.4x of original block_size
        scale = float(torch.empty(1).uniform_(0.6, 1.4))
        half_size = block_size * scale / 2.0

        x_min = center_xy[0] - half_size
        x_max = center_xy[0] + half_size
        y_min = center_xy[1] - half_size
        y_max = center_xy[1] + half_size

        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] < y_max)
        )

        if mask.sum() < min_points_per_block:
            return points, labels  # fallback
        return points[mask], labels[mask]

    @staticmethod
    def _random_xy_cutout(points, labels, keep_ratio=0.7):
        """
        Randomly keep only a subset of points (spatial dropout / cutout).
        """
        N = points.shape[0]
        if N < 50:
            return points, labels
        num_keep = int(N * keep_ratio)
        choice = torch.randperm(N)[:num_keep]
        return points[choice], labels[choice]

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data_blocks)

    def __getitem__(self, idx):
        # Base block (already windowed)
        points, labels = self.data_blocks[idx]  # points: [M, 3], labels: [M]

        # === Extra spatial augmentations: train/val only (block-level) ===
        if self.split in ("train", "val") and self.use_augmentation:
            # Optional multi-scale random crop on the block
            points, labels = self._multi_scale_random_crop(
                points,
                labels,
                block_size=self.block_size,
                min_points_per_block=self.min_points_per_block,
            )

        if self.split == "train" and self.use_augmentation:
            # Random cutout / point-level spatial dropout
            if torch.rand(1).item() < 0.5:
                points, labels = self._random_xy_cutout(
                    points, labels, keep_ratio=0.7
                )

        # === Sampling to fixed num_points ===
        num_pts = points.shape[0]
        if num_pts >= self.num_points:
            if self.use_fps:
                choice = PointCloudProcessor.farthest_point_sampling(
                    points[:, :3], self.num_points
                )
            else:
                choice = torch.randperm(num_pts)[: self.num_points]
        else:
            extra = torch.randint(num_pts, (self.num_points - num_pts,))
            choice = torch.cat([torch.arange(num_pts), extra], dim=0)

        points = points[choice]
        labels = labels[choice]

        # === (Optional) normalization ===
        if self.use_normization:
            mean = torch.mean(points, dim=0)
            std = torch.std(points, dim=0) + 1e-7
            points = (points - mean) / std
        else:
            mean = torch.zeros(points.shape[1])
            std = torch.ones(points.shape[1])

        # === Point-level augmentation (rotation, jitter, etc.) ===
        if self.use_augmentation and self.split == "train":
            points, labels = self.augmentor(points, labels)

        return points, labels, mean, std


# ----------------------------------------------------------------------
# Optional helpers
# ----------------------------------------------------------------------
def print_ply_file_counts(dataset_path):
    """
    Print number of points in each PLY file per split.
    """
    dataset_path = Path(dataset_path)
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        print(f"\nChecking PLY files in: {split_dir}")
        if not split_dir.exists():
            print("  (missing)")
            continue
        for ply_path in sorted(split_dir.glob("*.ply")):
            plydata = PlyData.read(str(ply_path))
            num_pts = plydata["vertex"].count
            print(f"  {ply_path.name}: {num_pts} points")
    print()


def print_sampled_class_distribution(dataset: PreprocessedDataset, name: str):
    """
    Iterate through the dataset using __getitem__ and count labels AFTER
    sampling to num_points (and after augmentation if enabled).
    Use this for debugging, not every epoch (can be slow).
    """
    label_counts = torch.zeros(dataset.num_classes, dtype=torch.long)

    for idx in range(len(dataset)):
        points, labels, mean, std = dataset[idx]  # labels: [num_points]
        labels = labels.view(-1)
        valid = (labels >= 0) & (labels < dataset.num_classes)
        labels = labels[valid]

        unique, counts = torch.unique(labels, return_counts=True)
        label_counts[unique] += counts

    total = label_counts.sum().item()
    print(f"\n[{name}] class distribution AFTER num_points sampling:")
    print(f"  Total sampled points: {total}")
    for c in range(dataset.num_classes):
        count = label_counts[c].item()
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"    Class {c}: {count} pts ({pct:.2f}%)")
    print()


# ----------------------------------------------------------------------
# DataLoader factory with WeightedRandomSampler and summary prints
# ----------------------------------------------------------------------
def get_dataloaders(config):
    # PLY-level sanity check
    # print_ply_file_counts(config.dataset_path)

    # ---------------- Train dataset ----------------
    train_dataset = PreprocessedDataset(
        dataset_path=config.dataset_path,
        split="train",
        num_points=config.num_points,
        block_size=config.block_size,
        stride=config.stride,  # overlap control for grid mode
        num_classes=config.num_classes,
        use_augmentation=config.use_augmentation,
        augmentation_args=getattr(config, "augmentation_args", None),
        use_normization=config.use_normization,
        use_fps=getattr(config, "use_fps", True),
        use_multi_view_blocks=getattr(config, "use_multi_view_blocks", True),
        random_shape_config=getattr(config, "random_shape_config", None),
        multi_scale_block_sizes=getattr(
            config, "multi_scale_block_sizes", (1.0, 100.0)
        ),
        load=getattr(config, "load_blocks", False),  # NEW
    )

    # ---- block-level sampling weights based on class imbalance ----
    class_weights = train_dataset.class_weights  # higher for rare classes
    block_weights = []
    for _, lbls in train_dataset.data_blocks:
        unique, _ = torch.unique(lbls, return_counts=True)
        w = class_weights[unique].max()  # emphasize blocks with rare classes
        block_weights.append(float(w))

    sampler = WeightedRandomSampler(
        weights=block_weights,
        num_samples=len(block_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,      # sampler instead of shuffle=True
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---------------- Val dataset ----------------
    val_dataset = PreprocessedDataset(
        dataset_path=config.dataset_path,
        split="val",
        num_points=config.num_points,
        block_size=config.block_size,
        stride=config.block_size,  # non-overlapping tiling for stability
        num_classes=config.num_classes,
        use_augmentation=False,    # typically no aug on val
        use_normization=config.use_normization,
        use_fps=getattr(config, "use_fps", True),
        use_multi_view_blocks=False,
        load=getattr(config, "load_blocks", False),  # NEW (optional)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ---------------- Test dataset ----------------
    test_dataset = PreprocessedDataset(
        dataset_path=config.dataset_path,
        split="test",
        num_points=config.num_points,
        block_size=config.block_size,
        stride=config.block_size,  # non-overlapping by default for eval
        num_classes=config.num_classes,
        use_augmentation=False,
        use_normization=config.use_normization,
        use_fps=getattr(config, "use_fps", True),
        use_multi_view_blocks=False,
        load=getattr(config, "load_blocks", False),  # NEW (optional)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ==== Summary logs (kept commented) ====
    # total_train_blocks = len(train_dataset)
    # total_val_blocks = len(val_dataset)
    # total_test_blocks = len(test_dataset)
    #
    # train_batches = len(train_loader)
    # val_batches = len(val_loader)
    # test_batches = len(test_loader)
    #
    # print("\n" + "=" * 80)
    # print(f"📊 Dataset Split Summary for {config.dataset_name}")
    # print(f"Train  Blocks (Before Aug): {total_train_blocks}")
    # print(f"Val    Blocks (Before Aug): {total_val_blocks}")
    # print(f"Test   Blocks:              {total_test_blocks}")
    # print("-" * 80)
    # if config.use_augmentation:
    #     print("🔁 Data augmentation is ENABLED for train.")
    #     print("   Train uses multi-view blocks (grid + random shapes + multi-scale).")
    # else:
    #     print("❌ Data augmentation is DISABLED.")
    # print("-" * 80)
    # print(f"🧮 Batch Counts:")
    # print(f"  Train loader batches : {train_batches}")
    # print(f"  Val loader batches   : {val_batches}")
    # print(f"  Test loader batches  : {test_batches}")
    # print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader, train_dataset.class_weights
