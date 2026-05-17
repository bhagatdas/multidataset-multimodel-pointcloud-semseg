# models/pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import sample_and_group, sample_and_group_all, index_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # +3 for xyz coordinates
        last_channel = in_channel + 3
        layers = []
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz, points):
        # xyz: [B, N, 3], points: [B, N, C] or None
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, _ = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_points: [B, S, nsample, C']
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C', nsample, S]
        new_points = self.mlp_convs(new_points)      # [B, C_out, nsample, S]
        new_points = torch.max(new_points, 2)[0]     # [B, C_out, S]
        new_points = new_points.permute(0, 2, 1)     # [B, S, C_out]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Feature Propagation from xyz2 -> xyz1

        xyz1: [B, N, 3]
        xyz2: [B, S, 3]
        points1: [B, N, C1] or None
        points2: [B, S, C2]
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  # [B, N, C2]
        else:
            # [B, N, S]
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # index_points(points2, idx): [B, N, 3, C2]
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1),
                dim=2,
            )  # [B, N, C2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, C1+C2]
        else:
            new_points = interpolated_points  # [B, N, C2]

        new_points = new_points.permute(0, 2, 1)  # [B, C, N]
        new_points = self.mlp_convs(new_points)   # [B, C', N]
        new_points = new_points.permute(0, 2, 1)  # [B, N, C']
        return new_points


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(1024, 0.05, 16, 3,   [32, 32, 64],    False)
        self.sa2 = PointNetSetAbstraction(256,  0.1, 16, 64,  [64, 64, 128],   False)
        self.sa3 = PointNetSetAbstraction(64,   0.2, 16, 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024], True)

        # Feature Propagation layers
        # Channels:
        #   l3_points: 256, l4_points: 1024  -> 256+1024 = 1280
        #   l2_points: 128, l3_points: 256   -> 128+256  = 384
        #   l1_points: 64,  l2_points: 256   -> 64+256   = 320
        #   l0_points: 3,   l1_points: 128   -> 3+128    = 131
        self.fp4 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384,  [256, 256])
        self.fp2 = PointNetFeaturePropagation(320,  [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 3, [128, 128, 128])

        # Output MLP
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        xyz: [B, N, 3]  (we treat coords as both xyz and initial features)
        Returns:
            logits: [B, N, num_classes]
        """
        l0_points = xyz          # [B, N, 3]  (features at level 0)
        l0_xyz = xyz[:, :, :3]   # [B, N, 3]

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)   # [B, 1024, 3], [B, 1024, 64]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # [B, 256, 3],  [B, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # [B, 64, 3],   [B, 64, 256]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)   # [B, 1, 3],    [B, 1, 1024]

        # Feature Propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [B, 64, 256]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 1024, 128]
        # ✅ use l0_points (xyz) as features so channels = 3 + 128 = 131
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # [B, N, 128]

        # Final per-point classification head
        x = l0_points.permute(0, 2, 1)        # [B, 128, N]
        x = F.relu(self.bn1(self.conv1(x)))   # [B, 128, N]
        x = self.drop1(x)
        x = self.conv2(x)                     # [B, num_classes, N]
        x = x.permute(0, 2, 1)                # [B, N, num_classes]
        return x
