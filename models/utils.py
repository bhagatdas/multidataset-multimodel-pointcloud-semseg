# models/utils.py
import torch

def square_distance(src, dst):
    """Calculate Euclidean distance between each two points.

    Args:
        src: [B, N, C]
        dst: [B, M, C]
    Returns:
        dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)    # [B, N, 1]
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)    # [B, 1, M]
    return dist


def index_points(points, idx):
    """
    Index points based on idx.

    Args:
        points: [B, N, C]
        idx:
            [B, S]           -> returns [B, S, C]
            [B, S, nsample]  -> returns [B, S, nsample, C]

    Returns:
        new_points: points gathered at idx.
    """
    device = points.device
    B = points.shape[0]

    # Make batch_indices broadcast to idx's shape
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(*view_shape).expand_as(idx)

    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """Furthest point sampling.

    Args:
        xyz: [B, N, 3]
        npoint: int

    Returns:
        centroids: [B, npoint] indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Group points using k-NN (nsample nearest neighbors).

    This avoids out-of-bounds issues from sentinel indices when
    no points fall within the given radius.

    Args:
        radius: float (ignored, kept for API compatibility)
        nsample: int
        xyz: [B, N, 3]    (all points)
        new_xyz: [B, S, 3]  (centroids)

    Returns:
        group_idx: [B, S, nsample] indices in [0, N-1]
    """
    # Compute pairwise squared distances: [B, S, N]
    sqrdists = square_distance(new_xyz, xyz)

    # Take indices of nsample nearest neighbors along N
    group_idx = sqrdists.argsort(dim=-1)[:, :, :nsample]  # [B, S, nsample]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """Sample and group points.

    Args:
        npoint: number of centroids S
        radius: radius for ball query
        nsample: number of neighbors
        xyz: [B, N, 3]
        points: [B, N, C'] or None

    Returns:
        new_xyz: [B, S, 3]
        new_points: [B, S, nsample, C'']  (C'' = 3 + C' if points not None)
        fps_idx: [B, S]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, S]
    new_xyz = index_points(xyz, fps_idx)          # [B, S, 3]

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, S, nsample]

    grouped_xyz = index_points(xyz, idx)                   # [B, S, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)         # [B, S, nsample, C']
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points, fps_idx


def sample_and_group_all(xyz, points):
    """Group all points (used in last SA layer)."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
