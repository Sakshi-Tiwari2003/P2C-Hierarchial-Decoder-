import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import MODELS
from models.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils import misc
from utils.logger import *

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global

# --- 4-Stage Hierarchical Decoder (Corrected) ---
class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim=1024, crop_point_num=2048):
        super().__init__()
        self.crop_point_num = crop_point_num

        # Feature compression
        self.fc0 = nn.Linear(latent_dim, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        # Stage generators: each stage generates points from previous stage
        # Stage 1: Coarse (32 points)
        self.fc0_1 = nn.Linear(2048, 32 * 3)
        # Stage 2: 64 points (2 per previous point)
        self.fc1_1 = nn.Linear(1024, 64 * 3)
        # Stage 3: 128 points (2 per previous point)
        self.fc2_1 = nn.Linear(512, 128 * 3)
        # Stage 4: Final (crop_point_num points, e.g., 2048)
        self.fc3_1 = nn.Linear(256, crop_point_num * 3)

    def forward(self, x):
        x0 = F.relu(self.fc0(x))
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))

        # Stage 1: Coarse (32 points)
        pc1 = self.fc0_1(x0).view(-1, 32, 3)

        # Stage 2: 64 points (2 per previous point)
        delta2 = self.fc1_1(x1).view(-1, 32, 2, 3)
        pc2 = pc1.unsqueeze(2) + delta2
        pc2 = pc2.view(-1, 64, 3)

        # Stage 3: 128 points (2 per previous point)
        delta3 = self.fc2_1(x2).view(-1, 64, 2, 3)
        pc3 = pc2.unsqueeze(2) + delta3
        pc3 = pc3.view(-1, 128, 3)

        # Stage 4: Final (crop_point_num points)
        delta4 = self.fc3_1(x3).view(-1, 128, self.crop_point_num // 128, 3)
        pc4 = pc3.unsqueeze(2) + delta4
        pc4 = pc4.view(-1, self.crop_point_num, 3)

        return pc4

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

# --- P2C Model with 4-Stage Decoder ---
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
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.manifold_constraint = ManifoldnessConstraint(support=config.support, neighborhood_size=config.neighborhood_size)
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.manifold_weight = config.manifold_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
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

        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat = self.encoder(rebuild_points.view(B, -1, 3))
        pred = self.generator(feat).contiguous()

        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()

        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()

        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)

        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty

    def forward(self, partial, n_points=None, record=False):
        feat = self.encoder(partial)
        pred = self.generator(feat).contiguous()
        return pred
