from __future__ import print_function
import open3d as o3d
import argparse
import torch
from numpy.ma.core import array

from model import MultiCON
from dataset import Synthesis, pairwise_distance
from data_model import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import GeodesicLoss, supervisedloss
import time
import copy
# from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate.')
    parser.add_argument('--gpu_id', default=0, type=str, help='Gpu ID.')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=True, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    args = parser.parse_args()
    return args

def compute_RTE(t_gt, t_pred):
    """计算相对平移误差 (RTE), 单位：与点云坐标同单位"""
    return np.linalg.norm(t_gt - t_pred)

def chamfer_distance(P, Q):
    """计算Chamfer Distance (CD)，输出平方距离"""
    # P, Q: shape (N, 3), (M, 3)
    tree_Q = cKDTree(Q)
    dist_PQ, _ = tree_Q.query(P, k=1)  # P 到 Q 的最近邻
    tree_P = cKDTree(P)
    dist_QP, _ = tree_P.query(Q, k=1)  # Q 到 P 的最近邻
    cd = np.mean(dist_PQ**2) + np.mean(dist_QP**2)
    return cd

def compute_RRE(R_gt, R_pred):
    """计算相对旋转误差 (RRE), 单位：度"""
    R_diff = R_gt.T @ R_pred
    cos_theta = (np.trace(R_diff) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定
    theta = np.arccos(cos_theta)
    RRE = np.degrees(theta)
    return RRE

def main():
    args = parse_args()
    model = MultiCON(args).cuda(args.gpu_id)

    test_loader = DataLoader(
        ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                   unseen=args.unseen, factor=args.factor),
        batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=8)

    model_path = ('./checkpoints/modelnet40_check/last_model.best.t7')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []
    all_distance = []

    R_all = []
    T_all = []
    cd_all = []

    iter = 0
    all_inner = []
    all_inner_half = []
    red = [1, 0, 0]
    green = [0, 1, 0]
    zi = [0, 0, 1]
    gray=[0.5,0.5,0.5]
    src_pcd = o3d.geometry.PointCloud()
    src_pcd_ori = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    target_pcd_ori = o3d.geometry.PointCloud()
    line_pcd = o3d.geometry.PointCloud()
    loss_number = 0
    times = []

    for src, target1, rotation, translation, I_gt in tqdm(test_loader):

        start = time.time()
        rotation_pred, translation_pred, _, target, _ = model(src.cuda(args.gpu_id), target1.cuda(args.gpu_id))
        end = time.time()
        times.append(end - start)

        rotations.append(rotation.numpy())
        translations.append(translation.numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())
        RRE = compute_RRE(rotation.squeeze(0),rotation_pred.squeeze(0).detach().cpu().numpy())
        RTE = np.linalg.norm(translation - translation_pred.detach().cpu().numpy())

        R_all.append(RRE)
        T_all.append(RTE)

        transforms_pred = torch.eye(4)
        transforms_pred[:3, :3] = rotation
        transforms_pred[:3, 3] = translation

        transforms_gt = torch.eye(4)
        transforms_gt[:3, :3] = rotation_pred.detach().cpu()
        transforms_gt[:3, 3] = translation_pred.detach().cpu()

        src_pcd.points = o3d.utility.Vector3dVector(src.squeeze(0).detach().cpu().T.numpy())

        target_pcd.points = o3d.utility.Vector3dVector(target.squeeze(0).detach().cpu().T.numpy())
        cd = chamfer_distance(np.array(src_pred.points),np.array(target_pcd.points))
        cd_all.append(cd)

        target_pcd.paint_uniform_color(red)
        src_pred.paint_uniform_color(green)
        src_pcd.paint_uniform_color(green)
        src_gt.paint_uniform_color(green)

        pc_ori = src_pcd + target_pcd
        o3d.io.write_point_cloud("./test_YYF_modelnet40/" + str(iter) + "_ori.ply", pc_ori)

        pc_pred = src_pred + target_pcd
        o3d.io.write_point_cloud("./test_YYF_modelnet40/" + str(iter) + "_pred.ply", pc_pred)
        #
        pc_gt = src_gt + target_pcd
        o3d.io.write_point_cloud("./test_YYF_modelnet40/" + str(iter) + "_gt.ply", pc_gt)

        with open("test_YYF_modelnet40/SSVCN_test_modelnet_best.txt", "a") as file:
            file.write(str(iter) + "_point-----" + "cd:" + str(cd) + "\n")
        iter += 1

    Mae = np.mean(all_distance)

    Mae_R = np.mean(R_all)
    Mae_T = np.mean(T_all)




if __name__ == '__main__':
    main()
