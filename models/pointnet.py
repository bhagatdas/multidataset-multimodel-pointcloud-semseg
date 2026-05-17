# models/pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, 3, N]
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity
        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(B, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, k, N]
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetSemSeg(nn.Module):
    """
    Vanilla PointNet for semantic segmentation.
    Input:  xyz [B, N, 3]
    Output: logits [B, N, num_classes]
    """
    def __init__(self, num_classes, feature_dim=3, use_feature_transform=True):
        super(PointNetSemSeg, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_feature_transform = use_feature_transform

        self.stn = STN3d()
        self.fstn = STNkd(k=64) if use_feature_transform else None

        # Point features
        self.conv1 = nn.Conv1d(feature_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Segmentation head
        # local_feat (64) + global_feat (1024) = 1088
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, num_classes, 1)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, xyz):
        """
        xyz: [B, N, 3]  (we assume only coordinates as input features here)
        """
        B, N, _ = xyz.shape

        # [B, 3, N]
        x = xyz.permute(0, 2, 1)

        # 3D alignment (T-Net)
        trans = self.stn(x)  # [B, 3, 3]
        x = torch.bmm(trans, x)  # [B, 3, N]

        # First feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, N]

        # Feature-space transform
        if self.use_feature_transform:
            trans_feat = self.fstn(x)        # [B, 64, 64]
            x = torch.bmm(trans_feat, x)     # [B, 64, N]
        else:
            trans_feat = None

        pointfeat = x                        # local 64-dim features

        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, N]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 128, N]
        x = self.bn5(self.conv5(x))          # [B, 1024, N]

        # Global feature
        x = torch.max(x, 2, keepdim=True)[0]    # [B, 1024, 1]
        x = x.repeat(1, 1, N)                   # [B, 1024, N]

        # Concatenate local + global
        x = torch.cat([pointfeat, x], dim=1)    # [B, 1088, N]

        # Per-point segmentation head
        x = F.relu(self.bn6(self.conv6(x)))     # [B, 512, N]
        x = F.relu(self.bn7(self.conv7(x)))     # [B, 256, N]
        x = F.relu(self.bn8(self.conv8(x)))     # [B, 128, N]
        x = self.dropout(x)
        x = self.conv9(x)                       # [B, num_classes, N]

        x = x.permute(0, 2, 1)                  # [B, N, num_classes]
        return x, trans, trans_feat
