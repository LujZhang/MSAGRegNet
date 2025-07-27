import random

import numpy as np
import torch
import torch.utils.data
import open3d as o3d
import torch.nn as nn
from PIL.ImImagePlugin import split

from scipy.spatial.transform import Rotation as R

class PointCloudAugmentor:
    def __init__(self, rotation_range=np.pi / 4, translation_range=0.5, jitter_sigma=0.01, jitter_clip=0.05):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def random_transform(self):
        """生成一个随机4x4增广RT矩阵"""
        angles = np.random.uniform(0, self.rotation_range, size=3)
        R_mat = R.from_euler('xyz', angles).as_matrix()
        T_vec = np.random.uniform(-self.translation_range, self.translation_range, size=3)
        RT = np.eye(4)
        RT[:3, :3] = R_mat
        RT[:3, 3] = T_vec
        return RT

    def apply_transform(self, pc, RT):
        """应用4x4变换到点云"""
        N = pc.shape[0]
        pc_h = np.hstack([pc, np.ones((N, 1))])  # (N, 4)
        pc_transformed = (RT @ pc_h.T).T[:, :3]
        return pc_transformed

    def jitter(self, pc):
        """加抖动噪声"""
        noise = np.clip(np.random.normal(0, self.jitter_sigma, pc.shape), -self.jitter_clip, self.jitter_clip)
        return pc + noise

    def augment(self, source, target, RT_gt):
        """
        增强流程：
        - 对source和target施加同一RT_aug
        - 返回 source_aug, target_aug, 新的RT_new = RT_aug @ RT_gt
        """
        RT_aug = self.random_transform()

        source_aug = self.apply_transform(source, RT_aug)

        source_aug = self.jitter(source_aug)
        target_aug = self.jitter(target)

        RT_new = RT_gt @ np.linalg.inv(RT_aug)

        return source_aug, target_aug, RT_new

class Pose3d(torch.utils.data.Dataset):
    def __init__(self, file_path, split = 'train', transformations=None):
        self.file_path = file_path
        self.transformations = transformations
        self.num_point = 1024

        array = np.load(self.file_path)
        self.pc = array['PC']
        self.length = len(self.pc)

        self.ct_array = array['CT']
        self.gt_array = array['GT']
        self.data_aug = PointCloudAugmentor()
        self.split = split

        # self.landmark_array = array['Landmark']

    def __getitem__(self, index):

        pc = self.pc[index].astype(np.float32)
        ct = self.ct_array[index // 100]

        gt = self.gt_array[index].astype(np.float32)
        gt = np.linalg.inv(gt)

        pc = pc[:, :3]

        if self.split == 'train' and random.randint(0, 1):
            pc,ct,gt = self.data_aug.augment(pc,ct,gt)

            pointcloud_pc = o3d.geometry.PointCloud()
            pointcloud_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
            pointcloud_pc.transform(gt)

            dist_map = torch.sqrt(pairwise_distance(torch.FloatTensor(pointcloud_pc.points), torch.FloatTensor(np.array(ct))))
            Ig = torch.zeros_like(dist_map)
            knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
            pc_index = torch.arange(1024)
            Ig[pc_index, knn_indices] = 1

            return torch.FloatTensor(pc[:, :3].T), torch.FloatTensor(ct.T), torch.FloatTensor(
                gt[:3, :3]), torch.FloatTensor(gt[:3, 3]),Ig
        else:
            pointcloud_pc = o3d.geometry.PointCloud()
            pointcloud_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
            pointcloud_pc.transform(gt)

            dist_map = torch.sqrt(pairwise_distance(torch.FloatTensor(np.array(pointcloud_pc.points)), torch.FloatTensor(np.array(ct))))
            Ig = torch.zeros_like(dist_map)
            knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
            pc_index = torch.arange(1024)
            Ig[pc_index, knn_indices] = 1
            return torch.FloatTensor(pc[:, :3].T), torch.FloatTensor(ct.T), torch.FloatTensor(gt[:3, :3]), torch.FloatTensor(gt[:3, 3]),Ig

    def __len__(self):
        return self.length

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances