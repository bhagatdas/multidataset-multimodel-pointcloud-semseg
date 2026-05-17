# models/pointnet2msg.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import index_points, farthest_point_sample, query_ball_point
from .pointnet2 import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        Multi-Scale Grouping (MSG) Set Abstraction.

        Args:
            npoint: int, number of centroids (S)
            radius_list: list[float], radii per scale (kept for API compatibility;
                         your query_ball_point is k-NN based)
            nsample_list: list[int], number of neighbors per scale
            in_channel: int, input feature dimension (excluding xyz)
            mlp_list: list[list[int]], MLP channels per scale, e.g.
                      [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        """
        super(PointNetSetAbstractionMSG, self).__init__()
        assert len(radius_list) == len(nsample_list) == len(mlp_list)

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.mlps = nn.ModuleList()
        for mlp in mlp_list:
            last_channel = in_channel + 3  # +3 for xyz
            layers = []
            for out_channel in mlp:
                layers.append(nn.Conv2d(last_channel, out_channel, 1))
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(nn.ReLU())
                last_channel = out_channel
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, xyz, points):
        """
        Args:
            xyz:    [B, N, 3]
            points: [B, N, C] or None

        Returns:
            new_xyz:    [B, S, 3]
            new_points: [B, S, sum_k C_out_k]
        """
        B, N, _ = xyz.shape

        # 1) Sample centroids by FPS
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, S]
        new_xyz = index_points(xyz, fps_idx)               # [B, S, 3]
        S = self.npoint

        if points is None:
            points = xyz  # treat xyz as initial features

        new_points_list = []

        # 2) Multi-scale grouping
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlps):
            # Your query_ball_point uses k-NN; radius is kept for API compat.
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, S, nsample]

            grouped_xyz = index_points(xyz, group_idx)       # [B, S, nsample, 3]
            grouped_points = index_points(points, group_idx) # [B, S, nsample, C]

            # Relative coordinates
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)   # [B, S, nsample, 3]

            # Concatenate coords + features
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, C+3]

            # [B, S, nsample, C'] -> [B, C', nsample, S]
            new_points = new_points.permute(0, 3, 2, 1)
            new_points = mlp(new_points)                       # [B, C_out, nsample, S]
            new_points = torch.max(new_points, 2)[0]           # [B, C_out, S]
            new_points = new_points.permute(0, 2, 1)           # [B, S, C_out]

            new_points_list.append(new_points)

        # 3) Concatenate multi-scale features
        new_points = torch.cat(new_points_list, dim=-1)        # [B, S, sum_k C_out_k]
        return new_xyz, new_points


class PointNet2MSGSemSeg(nn.Module):
    """
    PointNet++ MSG for semantic segmentation.

    Input:
        xyz: [B, N, 3]
    Output:
        logits: [B, N, num_classes]
    """
    def __init__(self, num_classes):
        super(PointNet2MSGSemSeg, self).__init__()

        # ------- MSG Set Abstraction layers (indoor-style example) -------

        # Level 1: npoint = 512
        # Output channels = 64 + 128 + 128 = 320
        self.sa1 = PointNetSetAbstractionMSG(
            npoint=512,
            radius_list=[0.05, 0.1, 0.2],
            nsample_list=[32, 64, 128],
            in_channel=3,   # treat xyz as initial features (C=3)
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ],
        )

        # Level 2: npoint = 128
        # Output channels = 128 + 256 + 256 = 640
        self.sa2 = PointNetSetAbstractionMSG(
            npoint=128,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=320,
            mlp_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ],
        )

        # Level 3: global abstraction (reuse SSG PointNetSetAbstraction)
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=640,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        # -> l3_points: [B, 1, 1024]

        # ------- Feature Propagation (reuse your FP module) -------
        # Channels:
        #   l2_points: 640, l3_points: 1024 -> 640 + 1024 = 1664
        #   l1_points: 320, l2_points': 512 -> 320 + 512  = 832
        #   l0_points: 3,   l1_points': 256 -> 3 + 256    = 259
        self.fp3 = PointNetFeaturePropagation(1664, [512, 512])
        self.fp2 = PointNetFeaturePropagation(832,  [512, 256])
        self.fp1 = PointNetFeaturePropagation(259,  [256, 128, 128])

        # ------- Output MLP -------
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3]

        Returns:
            logits: [B, N, num_classes]
        """
        l0_xyz = xyz
        l0_points = xyz  # use xyz as initial features

        # ----- Set Abstraction -----
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)   # [B, 512, 3], [B, 512, 320]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # [B, 128, 3], [B, 128, 640]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # [B, 1, 3],   [B, 1, 1024]

        # ----- Feature Propagation -----
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 128, 512]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 512, 256]
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # [B, N, 128]

        # ----- Final per-point head -----
        x = l0_points.permute(0, 2, 1)                 # [B, 128, N]
        x = F.relu(self.bn1(self.conv1(x)))            # [B, 128, N]
        x = self.drop1(x)
        x = self.conv2(x)                              # [B, num_classes, N]
        x = x.permute(0, 2, 1)                         # [B, N, num_classes]
        return x
