import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.build import MODELS
from models.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils import misc
from utils.logger import *


# --- PointNet++ Set Abstraction ---
class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input point positions, [B, N, 3]
            points: input point features, [B, N, C]
        Return:
            new_xyz: sampled point positions, [B, S, 3]
            new_points: corresponding features, [B, S, C']
        """
        B, N, C = xyz.shape

        if self.group_all:
            # Global pooling
            new_xyz = xyz[:, 0:1, :]  # [B, 1, 3]
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # Farthest Point Sampling
            new_xyz = pointops.fps(xyz, self.npoint)  # [B, npoint, 3]
            # KNN grouping
            idx = pointops.knn(new_xyz, xyz, self.nsample)[0]  # [B, npoint, nsample]
            grouped_xyz = pointops.index_points(xyz, idx)  # [B, npoint, nsample, 3]
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # normalize

            if points is not None:
                grouped_points = pointops.index_points(points, idx)  # [B, npoint, nsample, C]
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        # [B, npoint, nsample, C] -> [B, C, nsample, npoint]
        grouped_points = grouped_points.permute(0, 3, 2, 1)

        # Apply shared MLP (Conv2d + BN + ReLU)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling across local region
        new_points = torch.max(grouped_points, dim=2)[0]  # [B, C', npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C']

        return new_xyz, new_points


# --- Encoder (PointNet++ Based) ---
class Encoder(nn.Module):
    """
    PointNet++ based Encoder for global feature extraction
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        # PointNet++ Set Abstraction Layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[32, 32, 64], group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )

        # Global feature aggregation
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, feat_dim], group_all=True
        )

    def forward(self, x):
        """
        Input:
            x: [B, N, 3] - input point cloud
        Output:
            feature_global: [B, feat_dim] - global feature vector
        """
        xyz = x
        points = None

        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        xyz, points = self.sa3(xyz, points)
        xyz, points = self.sa4(xyz, points)

        feature_global = points.squeeze(1)  # [B, feat_dim]
        return feature_global


# --- Hierarchical Decoder ---
class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim=1024, crop_point_num=2048):
        super().__init__()
        self.crop_point_num = crop_point_num
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = nn.Conv1d(512, 512, 1)
        self.conv1_2 = nn.Conv1d(512, 256, 1)
        self.conv1_3 = nn.Conv1d(256, int((crop_point_num * 3) / 128), 1)

        self.conv2_1 = nn.Conv1d(128, 6, 1)

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x_3 = F.relu(self.fc3(x_2))

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.view(-1, 64, 3)

        pc2_feat = F.relu(self.fc2_1(x_2)).view(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat).permute(0, 2, 1).view(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz.unsqueeze(2) + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc3_feat = F.relu(self.fc1_1(x_1)).view(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat).permute(0, 2, 1)
        pc3_xyz = pc3_xyz.reshape(-1, 128, self.crop_point_num // 128, 3)
        pc3_xyz = pc2_xyz.unsqueeze(2) + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc3_xyz


# --- Manifoldness Constraint ---
class ManifoldnessConstraint(nn.Module):
    def __init__(self, support=8, neighborhood_size=32):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.support = support
        self.neighborhood_size = neighborhood_size

    def forward(self, xyz):
        normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)
        idx = pointops.knn(xyz, xyz, self.support)[0]
        neighborhood = pointops.index_points(normals, idx)
        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        penalty = 1 - cos_similarity
        penalty = penalty.std(-1).mean(-1)
        return penalty


# --- P2C Model ---
@MODELS.register_module()
class P2C(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.mask_ratio = config.mask_ratio
        self.feat_dim = config.feat_dim
        self.n_points = config.n_points
        self.nbr_ratio = config.nbr_ratio

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(self.feat_dim)
        self.generator = HierarchicalDecoder(latent_dim=self.feat_dim, crop_point_num=self.n_points)

        self.apply(self._init_weights)
        self._get_lossfnc_and_weights(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_lossfnc_and_weights(self, config):
        self.shape_criterion = ChamferDistanceL1()
        self.latent_criterion = lambda input, target: torch.log(torch.cosh(input - target)).mean()
        self.manifold_constraint = ManifoldnessConstraint(
            support=config.support, neighborhood_size=config.neighborhood_size
        )
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.manifold_weight = config.manifold_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups, center_groups = [], []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc + self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        nbrs, center = self.group_divider(pts)
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)

        # Shape Reconstruction Loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat = self.encoder(rebuild_points.view(B, -1, 3))
        pred = self.generator(feat).contiguous()

        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(
            rebuild_points.reshape(B, -1, 3), nbrs_pred
        ).mean()

        # Shape Matching Loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(
            rebuild_points.reshape(B, -1, 3), nbrs_pred
        ).mean()

        # Latent Reconstruction Loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)

        # Manifold Constraint
        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty
        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty

    def forward(self, partial, n_points=None, record=False):
        feat = self.encoder(partial)
        pred = self.generator(feat).contiguous()
        return pred
