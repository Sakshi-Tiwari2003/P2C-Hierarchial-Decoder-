import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.build import MODELS
from models.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils import misc
from utils.misc import fps



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
        pc1 = pc1_feat.view(-1, 64, 3)

        pc2_feat = F.relu(self.fc2_1(x_2)).view(-1, 128, 64)
        pc2 = self.conv2_1(pc2_feat).permute(0, 2, 1).view(-1, 64, 2, 3)
        pc2 = pc1.unsqueeze(2) + pc2
        pc2 = pc2.reshape(-1, 128, 3)

        pc3_feat = F.relu(self.fc1_1(x_1)).view(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3 = self.conv1_3(pc3_feat).permute(0, 2, 1).reshape(-1, 128, self.crop_point_num // 128, 3)
        pc3 = pc2.unsqueeze(2) + pc3
        pc3 = pc3.reshape(-1, self.crop_point_num, 3)

        return pc1, pc2, pc3

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

# --- P2C Model with Hierarchical Decoder ---
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
        pc1, pc2, pc3 = self.generator(feat)

        # Downsample GT using FPS
        
        fps_idx_64 = fps(rebuild_points.view(B, -1, 3), 64).long()
        fps_idx_128 = fps(rebuild_points.view(B, -1, 3), 128).long()
        gt_64 = pointops.index_points(rebuild_points.view(B, -1, 3), fps_idx_64)
        gt_128 = pointops.index_points(rebuild_points.view(B, -1, 3), fps_idx_128)



        loss1 = self.shape_recon_weight * self.shape_criterion(pc1, gt_64).mean()
        loss2 = self.shape_matching_weight * self.shape_criterion(pc2, gt_128).mean()
        loss3 = self.shape_matching_weight * self.shape_criterion(pc3, rebuild_points.view(B, -1, 3)).mean()

        idx = pointops.knn(center_groups[2], pc3, self.group_size)[0]
        nbrs_pred = pointops.index_points(pc3, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)

        manifold_penalty = self.manifold_weight * self.manifold_constraint(pc3).mean()

        total_loss = loss1 + loss2 + loss3 + latent_recon_loss + manifold_penalty
        return total_loss, loss1, loss2, loss3, latent_recon_loss, manifold_penalty

    def forward(self, partial, n_points=None, record=False):
        feat = self.encoder(partial)
        pc1, pc2, pc3 = self.generator(feat)
        return pc1, pc2, pc3
