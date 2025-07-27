#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import torch
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d
# Part of the code is referred from: https://github.com/charlesq34/pointnet

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


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/ModelNet')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data/ModelNet')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def consistent_pc(pc1, pc2):
    num_points, dim = pc1.shape  # 1024,3
    Id_1 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024
    Id_2 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024

    pc_xyz1 = Id_1 @ pc1  # 1024,3
    pc_xyz2 = Id_2 @ pc2  # 1024,3
    I_gt = Id_1 @ Id_2.T  # src = id1*id2^T*tgt = Ig*tgt # 1024,1024

    return pc_xyz1, pc_xyz2, I_gt


def random_partial_sample(pc1, pc2, num_sample):
    num_points, dim = pc1.shape  # 1024,3

    Id_1 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024
    Id_2 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024

    pc_xyz1 = Id_1 @ pc1  # 1024,3
    pc_xyz2 = Id_2 @ pc2  # 1024,3
    I_gt = Id_1 @ Id_2.T  # src = id1*id2^T*tgt = Ig*tgt # 1024,1024

    select1 = np.random.permutation(num_points)[:num_sample]
    select2 = np.random.permutation(num_points)[:num_sample]

    pointcloud1 = pc_xyz1[select1]  # num_sample,3
    pointcloud2 = pc_xyz2[select2]  # num_sample,3

    I_gt_temp = I_gt[select1].T
    I_gt = I_gt_temp[select2].T

    return pointcloud1, pointcloud2, I_gt


def farthest_partial_sample(pointcloud1, pointcloud2, num_subsampled_points=768):
    num_points = pointcloud1.shape[0]
    Id_1 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024
    Id_2 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024

    pc_xyz1 = Id_1 @ pointcloud1  # 1024,3
    pc_xyz2 = Id_2 @ pointcloud2  # 1024,3
    I_gt = Id_1 @ Id_2.T  # src = id1*id2^T*tgt = Ig*tgt # 1024,1024

    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pc_xyz1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pc_xyz2)
    random_p2 = random_p1  # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))

    pointcloud1 = pc_xyz1[idx1]  # num_sample,3
    pointcloud2 = pc_xyz2[idx2]  # num_sample,3

    I_gt_temp = I_gt[idx1].T
    I_gt = I_gt_temp[idx2].T

    return pointcloud1, pointcloud2, I_gt


def both_partial_sample(pc1, pc2, num_sample1, num_sample2):
    num_points, dim = pc1.shape  # 1024,3

    Id_1 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024
    Id_2 = np.eye(num_points)[np.random.permutation(num_points)]  # 1024,1024

    pc_xyz1 = Id_1 @ pc1  # 1024,3
    pc_xyz2 = Id_2 @ pc2  # 1024,3
    I_gt = Id_1 @ Id_2.T  # src = id1*id2^T*tgt = Ig*tgt # 1024,1024

    select1 = np.random.permutation(num_points)[:num_sample1]
    select2 = np.random.permutation(num_points)[:num_sample1]

    pointcloud1 = pc_xyz1[select1]  # num_sample,3
    pointcloud2 = pc_xyz2[select2]  # num_sample,3

    I_gt_temp = I_gt[select1].T
    I_gt = I_gt_temp[select2].T

    nbrs1 = NearestNeighbors(n_neighbors=num_sample2, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_sample2,))
    nbrs2 = NearestNeighbors(n_neighbors=num_sample2, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1  # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_sample2,))

    pointcloud1 = pointcloud1[idx1]
    pointcloud2 = pointcloud2[idx2]

    I_gt_temp = I_gt[idx1].T
    I_gt = I_gt_temp[idx2].T

    return pointcloud1, pointcloud2, I_gt


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        self.data_aug = PointCloudAugmentor()
        # self.data = self.data[self.label==20]
        # self.label = self.label[self.label==20]
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        partial = True
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = pointcloud1.T
        pointcloud2 = pointcloud2.T
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if partial:
            partial_format = 'farthest'
            num_partial_point_first = 512
            num_partial_point = 768

            if partial_format == 'random':
                pointcloud1, pointcloud2, I_gt = random_partial_sample(pointcloud1, pointcloud2, num_partial_point)
            if partial_format == 'farthest':
                pointcloud1, pointcloud2, I_gt = farthest_partial_sample(pointcloud1, pointcloud2, num_partial_point)
            if partial_format == 'both':
                pointcloud1, pointcloud2, I_gt = both_partial_sample(pointcloud1, pointcloud2, num_partial_point_first,
                                                                     num_partial_point)

        else:
            pointcloud1, pointcloud2, I_gt = consistent_pc(pointcloud1, pointcloud2)

        return pointcloud1.T.astype('float32'), pointcloud2.T.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), I_gt.astype('float32')

        # transforms = np.eye(4)
        # transforms[:3, :3] = R_ab
        # transforms[:3, 3] = translation_ab

        # if self.partition == 'train' and random.randint(0, 1):
        #     pointcloud1, pointcloud2, transforms = self.data_aug.augment(pointcloud1, pointcloud2, transforms)
        #     pointcloud_pc = o3d.geometry.PointCloud()
        #     pointcloud_pc.points = o3d.utility.Vector3dVector(pointcloud1[:, :3])
        #     pointcloud_pc.transform(transforms)

        #     dist_map = torch.sqrt(
        #         pairwise_distance(torch.FloatTensor(pointcloud_pc.points), torch.from_numpy(np.array(pointcloud2)).float()))
        #     Ig = torch.zeros_like(dist_map)
        #     knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
        #     pc_index = torch.arange(len(pointcloud_pc.points))
        #     Ig[pc_index, knn_indices] = 1

        #     return torch.FloatTensor(pointcloud1[:, :3].T), torch.FloatTensor(pointcloud2.T), torch.FloatTensor(
        #         transforms[:3, :3]), torch.FloatTensor(transforms[:3, 3]), Ig
        # else:
        #     pointcloud_pc = o3d.geometry.PointCloud()
        #     pointcloud_pc.points = o3d.utility.Vector3dVector(pointcloud1[:, :3])
        #     pointcloud_pc.transform(transforms)

        #     dist_map = torch.sqrt(
        #         pairwise_distance(torch.FloatTensor(pointcloud_pc.points), torch.from_numpy(np.array(pointcloud2)).float()))
        #     Ig = torch.zeros_like(dist_map)
        #     knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
        #     pc_index = torch.arange(len(pointcloud_pc.points))
        #     Ig[pc_index, knn_indices] = 1

        #     return torch.FloatTensor(pointcloud1[:, :3].T), torch.FloatTensor(pointcloud2.T), torch.FloatTensor(
        #         transforms[:3, :3]), torch.FloatTensor(transforms[:3, 3]), Ig


    def __len__(self):
        return self.data.shape[0]
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

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break