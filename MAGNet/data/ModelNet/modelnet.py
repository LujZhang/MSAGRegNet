"""Data loader for ModelNet40
"""
import argparse, os, torch, h5py, torchvision
from typing import List
import sys

import numpy as np
from plotly.graph_objs.layout.geo.projection import Rotation
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

import modelnet_transforms as Transforms

# def collate_pair(list_data):
#     """Collates data using a list, for tensors which are of different sizes
#     (e.g. different number of points). Otherwise, stacks them as per normal.
#     """
#
#     batch_sz = len(list_data)
#
#     # Collate as normal, other than fields that cannot be collated due to differing sizes,
#     # we retain it as a python list
#     to_retain_as_list = ['src_xyz', 'tgt_xyz', 'tgt_raw',
#                          'src_overlap', 'tgt_overlap',
#                          'correspondences',
#                          'src_path', 'tgt_path',
#                          'idx']
#     data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
#
#     return data

def get_train_datasets(root_path):
    train_categories = [line.rstrip('\n') for line in open('./data/ModelNet/modelnet40_half1.txt')]
    train_categories.sort()


    train_transforms, _ = get_transforms('crop', 45.0, 0.5,
                                                      1024, [0.7,0.7])
    train_transforms = torchvision.transforms.Compose(train_transforms)

    train_data = ModelNetHdf(root_path, subset='train', categories=train_categories,
                             transform=train_transforms)

    return train_data

def get_val_datasets(root_path):
    val_categories = [line.rstrip('\n') for line in open('./data/ModelNet/modelnet40_half1.txt')]
    val_categories.sort()

    _, val_transforms = get_transforms('crop', 45.0, 0.5,
                                                      1024, [0.7,0.7])
    val_transforms = torchvision.transforms.Compose(val_transforms)

    val_data = ModelNetHdf(root_path, subset='test', categories=val_categories,
                           transform=val_transforms)

    return val_data

def get_test_datasets(root_path):
    test_categories = [line.rstrip('\n') for line in open('./data/ModelNet/modelnet40_half2.txt')]
    test_categories.sort()

    _, test_transforms = get_transforms('crop', 45.0, 0.5,
                                                      1024, [0.7,0.7])
    test_transforms = torchvision.transforms.Compose(test_transforms)

    test_data = ModelNetHdf(root_path, subset='test', categories=test_categories,
                            transform=test_transforms)

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class ModelNetHdf(Dataset):
    def __init__(self, root: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            root (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._root = root
        self.n_in_feats = 1
        self.overlap_radius = 0.04

        if not os.path.exists(os.path.join(root)):
            self._download_dataset(root)

        with open(os.path.join(root, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(root, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._classes = categories
        else:
            categories_idx = None

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        self._transform = transform

    def __getitem__(self, item):
        sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        # Apply perturbation
        if self._transform:
            sample = self._transform(sample)

        corr_xyz = np.concatenate([
            sample['points_src'][sample['correspondences'][0], :3],
            sample['points_ref'][sample['correspondences'][1], :3]], axis=1)

        src_xyz = torch.from_numpy(sample['points_src'][:, :3])
        tgt_xyz = torch.from_numpy(sample['points_ref'][:, :3])
        transform = sample['transform_gt']

        dist_map = torch.sqrt(
            self.pairwise_distance(src_xyz, tgt_xyz))
        Ig = torch.zeros_like(dist_map)
        knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
        pc_index = torch.arange(len(src_xyz))
        Ig[pc_index, knn_indices] = 1
        # Ig = torch.from_numpy(Ig)
        rotation = torch.from_numpy(transform[:, :3])
        translation = torch.from_numpy(transform[:, 3])
        return src_xyz.T, tgt_xyz.T, rotation, translation, Ig

        # Transform to my format
        # sample_out = {
        #     'src_xyz': torch.from_numpy(sample['points_src'][:, :3]),
        #     'tgt_xyz': torch.from_numpy(sample['points_ref'][:, :3]),
        #     'tgt_raw': torch.from_numpy(sample['points_raw'][:, :3]),
        #     'src_overlap': torch.from_numpy(sample['src_overlap']),
        #     'tgt_overlap': torch.from_numpy(sample['ref_overlap']),
        #     'correspondences': torch.from_numpy(sample['correspondences']),
        #     'pose': torch.from_numpy(sample['transform_gt']),
        #     'idx': torch.from_numpy(sample['idx']),
        #     'corr_xyz': torch.from_numpy(corr_xyz),
        # }
        # return sample_out

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(root: str):
        os.makedirs(root, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(root)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]

    def pairwise_distance(self,
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