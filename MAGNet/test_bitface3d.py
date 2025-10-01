from __future__ import print_function
import open3d as o3d
import argparse
import torch
from numpy.ma.core import array

from model import MultiCON
from dataset_bitface3d import Synthesis, pairwise_distance
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import GeodesicLoss, supervisedloss
import time
# from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.', )
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.', )
    parser.add_argument('--lr', default=0.001, type=float, help='Base learning rate.', )
    parser.add_argument('--gpu_id', default=0, type=str, help='Gpu ID.')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
    args = parser.parse_args()
    return args

def compute_xyz_errors(R_Pre, R_Label, T_Pre, T_Label):
    """
    计算RT矩阵的误差，分别输出R的X、Y、Z轴旋转误差（度）和T的X、Y、Z轴平移误差。

    参数：
    R_Pre, R_Label: 3x3 旋转矩阵（预测 vs. 金标准）
    T_Pre, T_Label: 3x1 平移向量（预测 vs. 金标准）

    返回：
    (R_error_x, R_error_y, R_error_z, T_error_x, T_error_y, T_error_z)
    """
    # 计算旋转误差矩阵
    R_error = np.linalg.inv(R_Label) @ R_Pre

    # 提取旋转误差的 XYZ 轴欧拉角（单位：度）
    R_error_x, R_error_y, R_error_z = Rotation.from_matrix(R_error.squeeze(0)).as_euler('xyz', degrees=True)

    # 计算平移误差的 XYZ 分量
    T_error_x, T_error_y, T_error_z = (T_Pre - T_Label).flatten()

    return R_error_x, R_error_y, R_error_z, T_error_x, T_error_y, T_error_z

def main():
    args = parse_args()
    model = MultiCON(args).cuda(args.gpu_id)

    test_loader = DataLoader(
        Synthesis('./test_b_1024.npz'),
        batch_size=args.batch_size, shuffle=False, drop_last=False)

    model_path = ('./checkpoints/temp/last_model.best.t7')
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    input_size = [(1, 3, 1024), (1, 3, 1024)]
    # summary(model, input_size=input_size)
    # with open("test_ply_depth/SSVCN_test_log_best.txt", "a") as file:
    #         file.write(str(summary(model, input_size=input_size)))

    crit = GeodesicLoss()

    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []
    all_distance = []

    R_all = []
    T_all = []

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
    times = []

    for src, target, rotation, translation, I_gt, _ in tqdm(test_loader):

        start = time.time()
        rotation_pred, translation_pred, sssrc, sssrc_corr, _ = model(src.cuda(args.gpu_id), target.cuda(args.gpu_id))
        end = time.time()
        times.append(end - start)



        rotations.append(rotation.numpy())
        translations.append(translation.numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())

        R_dis = crit(rotation_pred, rotation.cuda(args.gpu_id))
        T_dis = torch.mean(torch.abs(translation_pred - translation.cuda(args.gpu_id)))

        R_all.append(R_dis.detach().cpu().numpy())
        T_all.append(T_dis.detach().cpu().numpy())

        transforms = torch.eye(4)
        transforms[:3, :3] = rotation_pred.detach().cpu()
        transforms[:3, 3] = translation_pred.detach().cpu()

        src_pcd.points = o3d.utility.Vector3dVector(src.squeeze(0).detach().cpu().T.numpy())
        src_pcd_ori.points = src_pcd.points
        # src_pcd.paint_uniform_color(green)
        src_pcd.transform(transforms.numpy())

        transforms[0, 3] = transforms[0, 3] + 0.1
        transforms[1, 3] = transforms[1, 3] - 0.15
        #
        line_pcd.points = o3d.utility.Vector3dVector(src.squeeze(0).detach().cpu().T.numpy())
        line_pcd.paint_uniform_color(green)
        #

        target_pcd.points = o3d.utility.Vector3dVector(sssrc_corr.squeeze(0).detach().cpu().T.numpy())
        # target_pcd.paint_uniform_color(red)

        target_pcd_ori.points = o3d.utility.Vector3dVector(target.squeeze(0).detach().cpu().T.numpy())
        target_pcd_ori.paint_uniform_color(gray)

        # all_points = src_pcd + target_pcd_ori
        # file_name =
        # o3d.io.write_point_cloud("./test_YYF_ply/" + str(iter) + "_combined_points.ply", all_points)
        #  o3d.visualization.draw_geometries(src_pcd,target_pcd)

        #the number of corresponding points is decrease after processing by model with SC2_PCR
        #must calculation again
        dist_map = torch.sqrt(
            pairwise_distance(torch.FloatTensor(src_pcd.points), torch.FloatTensor(target_pcd.points)))
        I_gt = torch.zeros_like(dist_map)
        knn_indices = dist_map.topk(k=1, dim=1, largest=False)[1].squeeze(-1)
        pc_index = torch.arange(1024)
        I_gt[pc_index, knn_indices] = 1

        #

        if iter == 501:
            line_pcd.transform(transforms.numpy())
            # 初始化
            window_name = "Point Cloud Registration -- VRNet-SC2"
            lines_red, lines_green = [], []
            points_red, points_green = [], []
            line_set_red, line_set_green = o3d.geometry.LineSet(), o3d.geometry.LineSet()
            red_line, green_line = [1, 0, 0], [0, 1, 0]

            # 使用目标点云构建KD树
            tar_tree = o3d.geometry.KDTreeFlann(src_pcd)

            # 遍历目标点云
            for index in range(0, len(target_pcd.points), 10):  # 步长为10加快处理
                point = target_pcd.points[index]
                [k, idx, _] = tar_tree.search_knn_vector_3d(point, 1)  # 最近邻
                nearest_point = src_pcd.points[idx[0]]
                line_point = line_pcd.points[idx[0]] if line_pcd else nearest_point
                distance = np.linalg.norm(np.array(point) - np.array(nearest_point))

                # 根据距离分类
                if distance < 0.01:
                    points_green.extend([point, line_point])
                    lines_green.append([len(points_green) - 2, len(points_green) - 1])
                else:
                    points_red.extend([point, line_point])
                    lines_red.append([len(points_red) - 2, len(points_red) - 1])

            # 调试输出
            print(f"Red Lines: {lines_red}")
            print(f"Green Lines: {lines_green}")

            # 构造红色线段
            line_set_red.points = o3d.utility.Vector3dVector(points_red)
            line_set_red.lines = o3d.utility.Vector2iVector(lines_red)
            line_set_red.paint_uniform_color(red_line)

            # 构造绿色线段
            line_set_green.points = o3d.utility.Vector3dVector(points_green)
            line_set_green.lines = o3d.utility.Vector2iVector(lines_green)
            line_set_green.paint_uniform_color(green_line)

            # 为源点云和目标点云设置颜色
            src_pcd.paint_uniform_color([1, 0.5, 0])  # 示例颜色：橙色
            target_pcd.paint_uniform_color([0, 0.5, 1])  # 示例颜色：蓝色

            # 可视化
            o3d.visualization.draw_geometries([src_pcd_ori, target_pcd, line_set_red, line_set_green])

            # o3d.visualization.draw_geometries([])

        tar_tree = o3d.geometry.KDTreeFlann(target_pcd)
        inner = 0
        inner_half = 0
        distances = []
        target_pcd_points=[]
        for i in range(np.asarray(src_pcd.points).shape[0]):
            point = np.asarray(src_pcd.points)[i]
            nearest_index = np.argmax(I_gt.squeeze(0).numpy()[i])
            nearest_point = np.asarray(target_pcd.points)[nearest_index]
            target_pcd_points.append(nearest_point)
            distance = np.linalg.norm(point - nearest_point)
            if distance < 0.01:
                if distance < 0.005:
                    inner_half += 1
                inner += 1
            distances.append(distance)
        target_pcd_points=np.array(target_pcd_points)

        norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))  # 归一化到[0, 1]
        cmap = plt.get_cmap('jet')  # 选择一个色图，例如 'jet'
        colors = cmap(norm_distances)[:, :3]
        # colors = np.array(colors)[:, :3]  # 只取 RGB 值
        src_pcd.colors = o3d.utility.Vector3dVector(colors)
        target_pcd_ori.paint_uniform_color(gray)
        # o3d.visualization.draw_geometries([src_pcd,target_pcd_ori])

        all_points = src_pcd + target_pcd_ori
        o3d.io.write_point_cloud("./test_ply_depth/" + str(iter) + "_combined_points_depth.ply", all_points)

        all_inner.append(inner / 1024)
        all_inner_half.append(inner_half / 1024)
        average_distance = np.mean(distances)
        all_distance.append(average_distance)
        with open("test_ply_depth/SSVCN_test_log_best.txt", "a") as file:
            file.write(str(iter) + "_point-----" + "distance:" + str(average_distance) + "\n")
        iter += 1

    Mae = np.mean(all_distance)

    Mae_R = np.mean(R_all)
    Mae_T = np.mean(T_all)
    inner_ratio = np.mean(all_inner)
    inner_ratio_half = np.mean(all_inner_half)
    fps = 500 / np.sum(times)
    print('Distance: %.5f' % (Mae))
    print('R_Distance: %.5f' % (Mae_R))
    print('T_Distance: %.5f' % (Mae_T))
    print('Inner ratio: %.5f' % (inner_ratio))
    print('Inner ratio 5 mm: %.5f' % (inner_ratio_half))
    print('FPS: %.5f' % (fps))
    with open("test_ply_depth/SSVCN_test_log_best.txt", "a") as file:
        file.write("Distance:" + str(Mae) + "\n"+"R_Distance:" + str(Mae_R) + "\n"+"T_Distance:" + str(Mae_T) + "\n"
                   "Inner ratio:" + str(inner_ratio) + "\n"+"'Inner ratio 5 mm:" + str(inner_ratio_half) + "\n"+"FPS:" + str(fps) + "\n")




if __name__ == '__main__':
    main()
