#!/usr/bin/env python
from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import MultiCON
from util import unsupervisedloss, GeodesicLoss, Myo3dKGCLoss, MyKGCLoss, Mysupervisedloss, supervisedloss
from dataset_bitface3d import Synthesis
from torchinfo import summary

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='PointCloud registration.')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate.')
    parser.add_argument('--gpu_id', default=0, type=str, help='Gpu ID.')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
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
    times=time.strftime('%y-%m-%d_%H:%M:%S', time.localtime())
    print(times)
    textio = IOStream('checkpoints/'+str(times)+'/run.log')


    train_loader = DataLoader(
        Synthesis('./train_b_1024.npz'),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        Synthesis('./test_b_1024.npz'),
        batch_size=args.batch_size, shuffle=False, drop_last=False)

    net = MultiCON(args).cuda(args.gpu_id)
    input_size = [(1, 3, 1024), (1, 3, 1024)]
    summary(net, input_size=input_size)
    crit = GeodesicLoss().cuda(args.gpu_id)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(opt, milestones=[], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)

    best_test_loss = np.inf

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_loss_1, train_loss_4, train_loss_5, train_loss_6, train_Igt_loss = train_one_epoch(args, net,
                                                                                                             train_loader,
                                                                                                             opt, crit)

        with torch.no_grad():
            test_loss, test_loss_1, test_loss_4, test_loss_5, test_loss_6, test_Igt_loss = test_one_epoch(args, net,
                                                                                                          test_loader,
                                                                                                          crit)

        train_loss_1_mae = np.mean(train_loss_1)
        train_loss_4_mae = np.mean(train_loss_4)
        train_loss_5_mae = np.mean(train_loss_5)
        train_loss_6_mae = np.mean(train_loss_6)
        # train_distance_mae = np.mean(train_distance)
        train_Igt_mae = np.mean(train_Igt_loss)

        test_loss_1_mae = np.mean(test_loss_1)
        test_loss_4_mae = np.mean(test_loss_4)
        test_loss_5_mae = np.mean(test_loss_5)
        test_loss_6_mae = np.mean(test_loss_6)
        # test_distance_mae = np.mean(test_distance)
        test_Igt_mae = np.mean(test_Igt_loss)
        torch.save(net.state_dict(), 'checkpoints/'+str(times)+'/last_model.best.t7')

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_loss1 = test_loss_1_mae
            best_test_loss4 = test_loss_4_mae
            best_test_loss5 = test_loss_5_mae
            best_test_loss6 = test_loss_6_mae
            # best_test_distance = test_distance_mae
            best_test_Igt = test_Igt_mae
            torch.save(net.state_dict(), 'checkpoints/'+str(times)+'/best_model.best.t7')

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, loss1: %f, loss4: %f, '
                      'loss5: %f, loss6: %f, Igt: %f'
                      % (epoch, train_loss, train_loss_1_mae, train_loss_4_mae, train_loss_5_mae, train_loss_6_mae,
                         train_Igt_mae))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, loss1: %f, loss4: %f, '
                      'loss5: %f, loss6: %f, Igt: %f'
                      % (epoch, test_loss, test_loss_1_mae, test_loss_4_mae, test_loss_5_mae, test_loss_6_mae,
                         test_Igt_mae))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, loss1: %f, loss4: %f, '
                      'loss5: %f, loss6: %f, Igt: %f'
                      % (epoch, best_test_loss, best_test_loss1, best_test_loss4, best_test_loss5, best_test_loss6,
                         best_test_Igt))
        gc.collect()


def test_one_epoch(args, net, test_loader, crit):
    net.eval()
    total_loss = 0
    num_examples = 0
    loss_1 = []
    loss_4 = []
    loss_5 = []
    loss_6 = []
    distance = []
    supervised_loss = []
    for src, target, rotation, translation, Ig, _, in tqdm(test_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_pred, translation_pred, src, corre_src, scores = net(src, target)

        # rt_pred = torch.cat((torch.cat((rotation_pred, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation_pred, torch.ones(batch_size, 1).cuda(args.gpu_id)),dim=1).unsqueeze(dim=2)), dim=2)
        # rt_gt = torch.cat((torch.cat((rotation, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation, torch.ones(batch_size, 1).cuda(args.gpu_id)),dim=1).unsqueeze(dim=2)), dim=2)

        loss1 = unsupervisedloss(src, corre_src, args)
        loss4 = Mysupervisedloss(rotation_pred, rotation)
        # loss4 = crit(rotation_pred, rotation)
        loss5 = Mysupervisedloss(translation_pred, translation)
        # loss6 = MyKGCLoss(src, target, rotation_pred, translation_pred)

        namta = 100.0
        suloss = supervisedloss(Ig, scores, args)
        suloss = namta * suloss
        # loss = 0.1 * loss1 + 2 * loss6
        loss = loss1 + suloss
        # dis = Myo3dKGCLoss(src, target, rt_pred)
        # distance.append(dis)

        loss_1.append(loss1.detach().cpu().numpy())
        loss_4.append(loss4.detach().cpu().numpy())
        loss_5.append(loss5.detach().cpu().numpy())
        supervised_loss.append(suloss.detach().cpu().numpy())
        # loss_6.append(loss6.detach().cpu().numpy())

        total_loss += loss.item() * batch_size
    return total_loss * 1.0 / num_examples, loss_1, loss_4, loss_5, loss_6, supervised_loss


def train_one_epoch(args, net, train_loader, opt, crit):
    net.train()
    num_examples = 0
    total_loss = 0
    loss_1 = []
    loss_4 = []
    loss_5 = []
    loss_6 = []
    distance = []
    supervised_loss = []

    for src, target, rotation, translation, Ig, _, in tqdm(train_loader):
        src = src.cuda(args.gpu_id)
        target = target.cuda(args.gpu_id)
        rotation = rotation.cuda(args.gpu_id)
        translation = translation.cuda(args.gpu_id)
        Ig = Ig.cuda(args.gpu_id)

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_pred, translation_pred, src, corre_src, scores = net(src, target)

        # rt_pred = torch.cat((torch.cat((rotation_pred, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                      torch.cat((translation_pred, torch.ones(batch_size, 1).cuda(args.gpu_id)),
        #                                dim=1).unsqueeze(dim=2)), dim=2)
        # rt_gt = torch.cat((torch.cat((rotation, torch.zeros(batch_size, 1, 3).cuda(args.gpu_id)), dim=1),
        #                    torch.cat((translation, torch.ones(batch_size, 1).cuda(args.gpu_id)), dim=1).unsqueeze(
        #                        dim=2)), dim=2)

        loss1 = unsupervisedloss(src, corre_src, args)
        loss4 = Mysupervisedloss(rotation_pred, rotation)
        # loss4 = crit(rotation_pred, rotation)
        loss5 = Mysupervisedloss(translation_pred, translation)
        # loss6 = MyKGCLoss(src, target, rotation_pred, translation_pred)

        namta = 100.0
        suloss = supervisedloss(Ig, scores, args)
        suloss = namta * suloss
        # loss = 0.1 * loss1 + 2 * loss6
        loss = loss1 + suloss

        # dis = Myo3dKGCLoss(src, target, rt_pred)
        # distance.append(dis)

        loss_1.append(loss1.detach().cpu().numpy())
        loss_4.append(loss4.detach().cpu().numpy())
        loss_5.append(loss5.detach().cpu().numpy())
        supervised_loss.append(suloss.detach().cpu().numpy())
        # loss_6.append(loss6.detach().cpu().numpy())

        loss.requires_grad_(True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples, loss_1, loss_4, loss_5, loss_6, supervised_loss


if __name__ == '__main__':
    main()
