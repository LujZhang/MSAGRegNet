#!/usr/bin/env python
from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import MultiCON
from util import unsupervisedloss, GeodesicLoss, Myo3dKGCLoss, MyKGCLoss, Mysupervisedloss, supervisedloss, \
    compute_recall_ir_dis
from dataset_pose import Pose3d
from dataset import Synthesis
# from test_modelnet40 import compute_RRE,compute_RTE
from torchinfo import summary

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def compute_RRE_torch(R_gt, R_pred):
    """RRE: Relative Rotation Error (degrees)"""
    R_diff = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    rre = torch.rad2deg(theta)
    return rre


def compute_RTE_torch(t_gt, t_pred):
    """RTE: Relative Translation Error (L2 distance)"""
    return torch.norm(t_gt - t_pred, dim=-1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.001, type=float, help='Base learning rate.')
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
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    args = parser.parse_args()
    return args


class IOStream:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_()
    times = time.strftime('%y_%m_%d_%H', time.localtime())
    print(times)
    textio = IOStream('checkpoints/' + str(times) + '/run.log')

    train_loader = DataLoader(
        Pose3d(file_path='data/angle_train.npz', split='train'),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(
        Pose3d(file_path='data/angle_test.npz', split='test'),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    # train_loader = DataLoader(
    #     Synthesis('./train_b_1024.npz'),
    #     batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(
    #     Synthesis('./test_b_1024.npz'),
    #     batch_size=args.batch_size, shuffle=False, drop_last=False)

    net = MultiCON(args).cuda(args.gpu_id)
    input_size = [(1, 3, 1024), (1, 3, 1024)]
    summary(net, input_size=input_size)
    crit = GeodesicLoss().cuda(args.gpu_id)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(opt, milestones=[], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    best_test_loss = np.inf
    textio.cprint('train begin')

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, un_loss, su_loss, rre, rte = train_one_epoch(args, net,
                                                                 train_loader,
                                                                 opt, crit)
        un_loss_mean = np.mean(un_loss)
        su_loss_mean = np.mean(su_loss)
        rre_mae = np.mean(rre)
        rte_mae = np.mean(rte)
        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, un_loss: %f, su_loss: %f, '
                      'rre: %f, rte: %f'
                      % (epoch, train_loss, un_loss_mean, su_loss_mean, rre_mae, rte_mae))

        with torch.no_grad():
            test_loss, un_loss, su_loss, rre, rte = test_one_epoch(args, net,
                                                                   test_loader,
                                                                   crit)

        un_loss_mean = np.mean(un_loss)
        su_loss_mean = np.mean(su_loss)
        rre_mae = np.mean(rre)
        rte_mae = np.mean(rte)
        torch.save(net.state_dict(), 'checkpoints/' + str(times) + '/last_model.best.t7')
        loss = rre_mae + rte_mae
        if best_test_loss >= loss:
            best_test_loss = loss
            best_un_loss = un_loss_mean
            best_su_loss = su_loss_mean
            best_rre = rre_mae
            best_rte = rte_mae
            torch.save(net.state_dict(), 'checkpoints/' + str(times) + '/best_model.best.t7')

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, un_loss: %f, su_loss: %f, '
                      'rre: %f, rte: %f'
                      % (epoch, test_loss, un_loss_mean, su_loss_mean, rre_mae, rte_mae))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, un_loss: %f, su_loss: %f, '
                      'rre: %f, rte: %f'
                      % (epoch, best_test_loss, best_un_loss, best_su_loss, best_rre, best_rte
                         ))
        gc.collect()


def test_one_epoch(args, net, test_loader, crit):
    net.eval()
    total_loss = 0
    num_examples = 0
    un_loss_list = []
    rre_list = []
    rte_list = []
    supervised_loss = []
    RR_list = []
    IR_list = []
    Pair_Dis_list = []

    for src, target, rotation, translation, Ig in tqdm(test_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_pred, translation_pred, src_pred, corre_src, scores = net(src, target)
        # print(RRE, RTE)

        # rt_pred = torch.cat((torch.cat((rotation_pred, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation_pred, torch.ones(batch_size, 1).cuda(args.gpu_id)),dim=1).unsqueeze(dim=2)), dim=2)
        # rt_gt = torch.cat((torch.cat((rotation, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation, torch.ones(batch_size, 1).cuda(args.gpu_id)),dim=1).unsqueeze(dim=2)), dim=2)

        un_loss = unsupervisedloss(src_pred, corre_src, args)

        RRE = compute_RRE_torch(R_gt=rotation, R_pred=rotation_pred)
        RTE = compute_RTE_torch(t_gt=translation, t_pred=translation_pred)
        # print(RRE, RTE)

        namta = 100.0
        suloss = supervisedloss(Ig, scores, args)
        suloss = namta * suloss
        # loss = 0.1 * loss1 + 2 * loss6
        loss = un_loss + suloss + RRE + RTE

        un_loss_list.append(un_loss.detach().cpu().numpy())
        supervised_loss.append(suloss.detach().cpu().numpy())
        rre_list.append(RRE.detach().cpu().numpy())
        rte_list.append(RTE.detach().cpu().numpy())


        total_loss += loss.item() * batch_size

        rre_list.append(RRE.detach().cpu().numpy())
        rte_list.append(RTE.detach().cpu().numpy())
        RR, IR, Pair_Dis = compute_recall_ir_dis(src, target, Ig, rotation_pred, translation_pred)
        RR_list.append(RR)
        IR_list.append(IR)
        Pair_Dis_list.append(Pair_Dis)
    print(f"Recall (GT match): {np.mean(RR_list):.4f}")
    print(f"Inner Ratio (NN dist): {np.mean(IR_list):.4f}")
    print(f"Average Pair Distance: {np.mean(Pair_Dis_list):.4f}")

    return total_loss * 1.0 / num_examples, un_loss_list, supervised_loss, rre_list, rte_list


def train_one_epoch(args, net, train_loader, opt, crit):
    net.train()
    num_examples = 0
    total_loss = 0
    un_loss_list = []
    rre_list = []
    rte_list = []
    supervised_loss = []

    RR_list = []
    IR_list = []
    Pair_Dis_list = []

    for src, target, rotation, translation, Ig in tqdm(train_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_pred, translation_pred, src_pred, corre_src, scores = net(src, target)

        # rt_pred = torch.cat((torch.cat((rotation_pred, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation_pred, torch.ones(batch_size, 1).cuda(args.gpu_id)),
        #                                dim=1).unsqueeze(dim=2)), dim=2)
        # rt_gt = torch.cat((torch.cat((rotation, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                    torch.cat((translation, torch.ones(batch_size, 1).cuda(args.gpu_id)), dim=1).unsqueeze(
        #                        dim=2)), dim=2)

        un_loss = unsupervisedloss(src_pred, corre_src, args)

        RRE = compute_RRE_torch(R_gt=rotation, R_pred=rotation_pred)
        RTE = compute_RTE_torch(t_gt=translation, t_pred=translation_pred)
        # print(RRE, RTE)

        namta = 100.0
        suloss = supervisedloss(Ig, scores, args)
        suloss = namta * suloss
        loss = un_loss + suloss


        un_loss_list.append(un_loss.detach().cpu().numpy())
        supervised_loss.append(suloss.detach().cpu().numpy())

        loss.requires_grad_(True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        rre_list.append(RRE.detach().cpu().numpy())
        rte_list.append(RTE.detach().cpu().numpy())
        RR,IR,Pair_Dis = compute_recall_ir_dis(src,target,Ig,rotation_pred,translation_pred)
        RR_list.append(RR)
        IR_list.append(IR)
        Pair_Dis_list.append(Pair_Dis)
    # loss_6.append(loss6.detach().cpu().numpy())
    print(f"Recall (GT match): {np.mean(RR_list):.4f}")
    print(f"Inner Ratio (NN dist): {np.mean(IR_list):.4f}")
    print(f"Average Pair Distance: {np.mean(Pair_Dis_list):.4f}")
    total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples, un_loss_list, supervised_loss, rre_list, rte_list


if __name__ == '__main__':
    main()
