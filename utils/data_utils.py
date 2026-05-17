import torch
from typing import Tuple, List, Optional, Dict

class PointCloudProcessor:
    """
    Point cloud utilities for preprocessing and augmentation (CUDA-compatible).
    """

    @staticmethod
    def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
        centroid = points.mean(dim=0)
        points = points - centroid
        max_dist = torch.sqrt(torch.sum(points ** 2, dim=1)).max()
        if max_dist > 0:
            points = points / max_dist
        return points

    @staticmethod
    def normalize_to_block(points: torch.Tensor, block_center: torch.Tensor) -> torch.Tensor:
        return points - block_center

    @staticmethod
    def random_rotation_z(points: torch.Tensor) -> torch.Tensor:
        theta = torch.rand(1, device=points.device) * 2 * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rot = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ], device=points.device).float()
        return torch.matmul(points, rot)

    @staticmethod
    def random_rotation_3d(points: torch.Tensor) -> torch.Tensor:
        angles = torch.rand(3, device=points.device) * 2 * torch.pi
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]),  torch.cos(angles[0])]
        ], device=points.device)
        Ry = torch.tensor([
            [ torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], device=points.device)
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]),  torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], device=points.device)
        R = torch.mm(Rz, torch.mm(Ry, Rx))
        return torch.matmul(points, R)

    @staticmethod
    def random_scale(points: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        scale = (scale_range[1] - scale_range[0]) * torch.rand(1, device=points.device) + scale_range[0]
        return points * scale

    @staticmethod
    def random_anisotropic_scale(points: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        scales = (scale_range[1] - scale_range[0]) * torch.rand(3, device=points.device) + scale_range[0]
        return points * scales

    @staticmethod
    def random_jitter(points: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
        jitter = torch.normal(mean=0, std=sigma, size=points.shape, device=points.device).float()
        jitter = torch.clamp(jitter, -clip, clip)
        return points + jitter

    @staticmethod
    def random_flip(points: torch.Tensor, axis: int = 0) -> torch.Tensor:
        if torch.rand(1, device=points.device) > 0.5:
            points = points.clone()
            points[:, axis] = -points[:, axis]
        return points

    @staticmethod
    def random_dropout(points: torch.Tensor, labels: torch.Tensor, max_dropout_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        dropout_ratio = torch.rand(1, device=points.device).item() * max_dropout_ratio
        num_points = len(points)
        num_keep = int(num_points * (1 - dropout_ratio))
        indices = torch.randperm(num_points, device=points.device)[:num_keep]
        return points[indices], labels[indices]

    @staticmethod
    def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
        N = points.shape[0]
        if num_samples >= N:
            return torch.arange(N, device=points.device)
        centroids = torch.zeros(num_samples, dtype=torch.long, device=points.device)
        distance = torch.ones(N, device=points.device) * 1e10
        farthest = torch.randint(0, N, (1,), device=points.device).item()
        for i in range(num_samples):
            centroids[i] = farthest
            centroid_point = points[farthest, :]
            dist = torch.sum((points - centroid_point) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance).item()
        return centroids

    @staticmethod
    def random_sampling(points: torch.Tensor, labels: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = len(points)
        if num_points >= num_samples:
            indices = torch.randperm(num_points, device=points.device)[:num_samples]
        else:
            indices_base = torch.arange(num_points, device=points.device)
            indices_extra = torch.randint(0, num_points, (num_samples - num_points,), device=points.device)
            indices = torch.cat([indices_base, indices_extra])
        return points[indices], labels[indices]

class DataAugmentation:
    """
    Data augmentation for point clouds (CUDA-optimized).
    """
    def __init__(self,
                 use_rotation=True,
                 use_scaling=True,
                 use_jitter=True,
                 use_flip=True,
                 use_dropout=False,
                 rotation_3d=True,
                 scale_range=(0.8, 1.2),
                 jitter_sigma=0.01,
                 jitter_clip=0.05,
                 dropout_ratio=0.2):
        self.use_rotation = use_rotation
        self.use_scaling = use_scaling
        self.use_jitter = use_jitter
        self.use_flip = use_flip
        self.use_dropout = use_dropout
        self.rotation_3d = rotation_3d
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.dropout_ratio = dropout_ratio
        self.processor = PointCloudProcessor()

    def __call__(self, points: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_rotation:
            if self.rotation_3d:
                points = self.processor.random_rotation_3d(points)
            else:
                points = self.processor.random_rotation_z(points)
        if self.use_scaling:
            points = self.processor.random_scale(points, self.scale_range)
        if self.use_jitter:
            points = self.processor.random_jitter(points, self.jitter_sigma, self.jitter_clip)
        if self.use_flip:
            points = self.processor.random_flip(points, axis=0)
        if self.use_dropout and labels is not None:
            points, labels = self.processor.random_dropout(points, labels, self.dropout_ratio)
        return points, labels

class BlockSplitter:
    """
    Split large point clouds into blocks for processing (CUDA-optimized).
    """
    def __init__(self, block_size: float = 50.0, stride: float = 25.0, min_points: int = 100):
        self.block_size = block_size
        self.stride = stride
        self.min_points = min_points

    def split(self, points: torch.Tensor, labels: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        blocks = []

        # Get bounding box (XY only for block splitting)
        min_coords = torch.min(points[:, :2], dim=0)[0]
        max_coords = torch.max(points[:, :2], dim=0)[0]

        x_range = max_coords[0] - min_coords[0]
        y_range = max_coords[1] - min_coords[1]
        x_steps = int(torch.ceil(x_range / self.stride).item())
        y_steps = int(torch.ceil(y_range / self.stride).item())

        for i in range(x_steps):
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

                if block_points.shape[0] < self.min_points:
                    continue

                # Block center for downstream normalization if needed
                block_center = torch.tensor([
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    block_points[:, 2].mean() if block_points.shape[0] > 0 else 0
                ], device=points.device)
                blocks.append((block_points, block_labels, block_center))

        return blocks

class DatasetStatistics:
    """
    Compute and analyze dataset statistics (CUDA-optimized).
    """
    @staticmethod
    def compute_class_distribution(labels: torch.Tensor, num_classes: int) -> Dict[str, any]:
        unique, counts = torch.unique(labels, return_counts=True)
        class_counts = torch.zeros(num_classes, device=labels.device)
        for u, c in zip(unique, counts):
            if u < num_classes:
                class_counts[u] = c
        total = class_counts.sum()
        class_percentages = class_counts / (total + 1e-10) * 100
        return {
            'class_counts': class_counts.cpu().tolist(),
            'class_percentages': class_percentages.cpu().tolist(),
            'total_points': int(total.item()),
            'num_classes_present': int((class_counts > 0).sum().item())
        }

    @staticmethod
    def compute_point_density(points: torch.Tensor) -> float:
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        volume = torch.prod(max_coords - min_coords)
        if volume > 0:
            return float(points.shape[0] / volume.item())
        else:
            return 0.0

    @staticmethod
    def compute_bounding_box(points: torch.Tensor) -> Dict[str, any]:
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        return {
            'min': min_coords.cpu().tolist(),
            'max': max_coords.cpu().tolist(),
            'center': center.cpu().tolist(),
            'size': size.cpu().tolist(),
            'volume': float(torch.prod(size).item())
        }

    @staticmethod
    def analyze_dataset(data_blocks: List[Tuple[torch.Tensor, torch.Tensor]],
                       num_classes: int,
                       class_names: List[str]) -> Dict[str, any]:
        all_labels = []
        all_points_count = []
        all_densities = []

        for points, labels in data_blocks:
            all_labels.append(labels)
            all_points_count.append(points.shape[0])
            if points.shape[0] > 0:
                min_coords = torch.min(points, dim=0)[0]
                max_coords = torch.max(points, dim=0)[0]
                volume = torch.prod(max_coords - min_coords + 1e-10)
                all_densities.append(float(points.shape[0] / volume.item()))

        all_labels = torch.cat(all_labels, dim=0)
        class_dist = DatasetStatistics.compute_class_distribution(all_labels, num_classes)

        block_stats = {
            'num_blocks': len(data_blocks),
            'avg_points_per_block': float(torch.mean(torch.tensor(all_points_count)).item()),
            'std_points_per_block': float(torch.std(torch.tensor(all_points_count)).item()),
            'min_points_per_block': int(min(all_points_count)),
            'max_points_per_block': int(max(all_points_count)),
            'avg_density': float(torch.mean(torch.tensor(all_densities)).item()) if all_densities else 0,
        }

        per_class_info = []
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f'class_{i}'
            per_class_info.append({
                'class_id': i,
                'class_name': class_name,
                'count': int(class_dist['class_counts'][i]),
                'percentage': float(class_dist['class_percentages'][i])
            })

        return {
            'overview': {
                'total_blocks': len(data_blocks),
                'total_points': class_dist['total_points'],
                'num_classes': num_classes,
                'classes_present': class_dist['num_classes_present']
            },
            'block_statistics': block_stats,
            'class_distribution': per_class_info,
            'summary': class_dist
        }
