import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import index_points, farthest_point_sample, query_ball_point
from .pointnet2 import PointNetFeaturePropagation


###############################################
# XConv with Residual + BatchNorm
###############################################
class XConv(nn.Module):
    def __init__(self, K, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Positional MLP produces K x K transformation matrix
        self.mlp_pos = nn.Sequential(
            nn.Linear(K * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, K * K),
        )

        # Feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, local_xyz, grouped_points):
        """
        local_xyz:      [B, S, K, 3]   (relative coords)
        grouped_points: [B, S, K, C_in]
        """
        B, S, K, _ = local_xyz.shape
        BS = B * S
        _, _, K2, C_in = grouped_points.shape

        # Ensure neighborhood size matches what layer was constructed with
        assert K == self.K and K2 == self.K, (
            f"XConv got K={K}, K2={K2}, but self.K={self.K}"
        )

        # [B, S, K, 3] -> [B*S, K*3]
        pos = local_xyz.reshape(BS, K * 3)

        # [B*S, K*3] -> [B*S, K*K]
        X = self.mlp_pos(pos).view(BS, K, K)
        X = F.softmax(X, dim=-1)

        # [B, S, K, C_in] -> [B*S, K, C_in]
        feats = grouped_points.reshape(BS, K, C_in)

        # [B*S, K, K] @ [B*S, K, C_in] -> [B*S, K, C_in]
        f_trans = torch.bmm(X, feats)

        # Aggregate over K dimension -> [B*S, C_in]
        f_agg, _ = torch.max(f_trans, dim=1)

        # Feature MLP -> [B*S, out_channels]
        out = self.mlp_feat(f_agg)

        # Residual connection if dims match
        if self.in_channels == self.out_channels:
            out = out + f_agg

        return out.view(B, S, self.out_channels)


###############################################
# PointCNN Layer with Dilated kNN
###############################################
class PointCNNLayer(nn.Module):
    def __init__(self, npoint, K, in_channels, out_channels,
                 hidden_channels=64, dilation=1):
        super().__init__()
        self.npoint = npoint       # number of centroids
        self.K = K                 # number of neighbors fed to XConv
        self.dilation = dilation
        self.eff_K = K * dilation  # number of neighbors to sample before dilation

        self.xconv = XConv(K, in_channels, out_channels, hidden_channels)

    def forward(self, xyz, points):
        """
        xyz:    [B, N, 3]
        points: [B, N, C_in] or None
        """
        B, N, _ = xyz.shape

        # Farthest point sampling
        if self.npoint is None or self.npoint >= N:
            new_xyz = xyz
            S = N
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
            new_xyz = index_points(xyz, fps_idx)               # [B, npoint, 3]
            S = self.npoint

        # Group neighbors (dilated kNN style)
        nsample = min(self.eff_K, N)  # cannot sample more than N points
        group_idx = query_ball_point(
            radius=None,
            nsample=nsample,
            xyz=xyz,
            new_xyz=new_xyz
        )  # [B, S, nsample]

        # Apply dilation by sub-sampling neighbor indices
        if self.dilation > 1:
            group_idx = group_idx[:, :, ::self.dilation]  # [B, S, K'] (K' ≈ nsample/dilation)

        # ---------- NEW: enforce exactly self.K neighbors ----------
        Kp = group_idx.shape[2]
        target_K = self.K

        if Kp > target_K:
            # Too many neighbors: keep first target_K
            group_idx = group_idx[:, :, :target_K]
        elif Kp < target_K:
            # Too few neighbors: pad by repeating the last index
            pad = group_idx[:, :, -1:].expand(B, S, target_K - Kp)
            group_idx = torch.cat([group_idx, pad], dim=2)
        # Now group_idx.shape[2] == self.K
        # -----------------------------------------------------------

        grouped_xyz = index_points(xyz, group_idx)      # [B, S, K, 3]
        if points is None:
            grouped_points = grouped_xyz                # [B, S, K, 3]
        else:
            grouped_points = index_points(points, group_idx)  # [B, S, K, C_in]

        local_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # [B, S, K, 3]

        # XConv expects exactly self.K neighbors
        new_points = self.xconv(local_xyz, grouped_points)  # [B, S, C_out]

        return new_xyz, new_points


###############################################
# Final PointCNN Semantic Segmentation Network
###############################################
class PointCNNSemSeg(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        self.layer1 = PointCNNLayer(
            npoint=1024,
            K=64,
            in_channels=in_channels,
            out_channels=64,
            hidden_channels=64,
            dilation=1
        )
        self.layer2 = PointCNNLayer(
            npoint=256,
            K=72,
            in_channels=64,
            out_channels=128,
            hidden_channels=128,
            dilation=1
        )
        self.layer3 = PointCNNLayer(
            npoint=64,
            K=80,
            in_channels=128,
            out_channels=256,
            hidden_channels=256,
            dilation=1
        )
        self.layer4 = PointCNNLayer(
            npoint=16,
            K=96,
            in_channels=256,
            out_channels=512,
            hidden_channels=512,
            dilation=1
        )

        # Feature propagation
        # After fp4: l3_points becomes 512-dim
        self.fp4 = PointNetFeaturePropagation(512 + 256, [512, 512])  # in: 768

        # After fp4, l3_points: 512 channels; l2_points: 128 -> 512 + 128 = 640
        self.fp3 = PointNetFeaturePropagation(512 + 128, [512, 256])  # in: 640, out: 256

        # After fp3, l2_points: 256; l1_points: 64 -> 256 + 64 = 320
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])   # in: 320, out: 128

        # After fp2, l1_points: 128; l0_points: in_channels -> in_channels + 128
        self.fp1 = PointNetFeaturePropagation(in_channels + 128, [128, 128, 128])

        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, feat=None):
        """
        xyz:  [B, N, 3]
        feat: [B, N, C_in] or None (if None, use xyz as features)
        """
        if feat is None:
            feat = xyz

        l0_xyz, l0_points = xyz, feat

        # Encoder
        l1_xyz, l1_points = self.layer1(l0_xyz, l0_points)    # [B, 1024, 64]
        l2_xyz, l2_points = self.layer2(l1_xyz, l1_points)    # [B, 256, 128]
        l3_xyz, l3_points = self.layer3(l2_xyz, l2_points)    # [B, 64, 256]
        l4_xyz, l4_points = self.layer4(l3_xyz, l3_points)    # [B, 16, 512]

        # Decoder / Feature Propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # -> [B, 64, 512]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # -> [B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # -> [B, 1024, 128]
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # -> [B, N, 128]

        # Segmentation head
        x = l0_points.permute(0, 2, 1)        # [B, 128, N]
        x = F.relu(self.bn1(self.conv1(x)))   # [B, 128, N]
        x = self.drop1(x)
        x = self.conv2(x)                     # [B, num_classes, N]

        return x.permute(0, 2, 1)             # [B, N, num_classes]
