import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import open3d as o3d


class MultiCON(nn.Module):
    def __init__(self, args):
        super(MultiCON, self).__init__()
        self.emb_dims = args.emb_dims
        self.args = args
        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        self.pointer = Transformer(args=args)
        self.head = SVDHead(args=args)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab, scr, src_corr, scores = self.head(src_embedding, tgt_embedding, src, tgt)
        # print(scores)
        rotation_ba = rotation_ab.transpose(2, 1).contiguous()

        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        return rotation_ab, translation_ab, scr, src_corr, scores

class Att_pooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Att_pooling, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x_att = self.conv(x)
        x_score = torch.sigmoid(x_att)
        x = x * x_score
        x = torch.sum(x,dim=-1)
        x = self.conv2(x)

        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(8 * 2, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(8 * 2, 64, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)


        # self.conv5 = nn.Conv1d(512, emb_dims, kernel_size=1, bias=False)


        self.conv = nn.Conv1d(512, emb_dims, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(emb_dims)
        #
        # self.conv5 = nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False)
        # self.bn5 = nn.BatchNorm1d(emb_dims)

        self.liner = nn.Conv1d(3, 8, kernel_size=1, bias=False)
        self.liner_bn = nn.BatchNorm1d(8)

        # self.bn5 = nn.BatchNorm2d(emb_dims)

        self.att_1 = Att_pooling(128, 256)

        self.att_2 = Att_pooling(128, 256)

        # self.att_3= Att_pooling(512, emb_dims)

    def forward(self, x):
        x = F.relu(self.liner_bn(self.liner(x)))

        x_1 = get_graph_feature(x,k=16)
        x_1 = F.relu(self.bn1(self.conv1(x_1)))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = self.att_1(x_1)

        x_2 = get_graph_feature(x,k=12)
        x_2 = F.relu(self.bn3(self.conv3(x_2)))
        x_2 = F.relu(self.bn4(self.conv4(x_2)))
        x_2 = self.att_2(x_2)

        x = F.relu(torch.concat([x_1,x_2],dim=1))

        x = F.relu(self.bn(self.conv(x)))


        # x = F.relu(self.bn5(self.conv5(x + x_skip)))
        # x = x.view(batch_size, -1, num_points)
        return x


def get_graph_feature(x, k=16, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.hidden_dim = 64
        self.to_q = nn.Linear(d_model, self.hidden_dim )
        self.to_k = nn.Linear(d_model, self.hidden_dim )
        self.to_v = nn.Linear(d_model, self.hidden_dim )
        self.linears = clones(nn.Linear(d_model, self.hidden_dim), 4)
        self.project = nn.Linear(self.hidden_dim,d_model)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query=self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value)

        # 2) Apply attention on all the projected vectors in batch.

        weights = torch.mul(torch.softmax(key, 1), value).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(query)
        Yt=torch.mul(Q_sig,weights)

        # 3) "Concat" using a view and apply a final linear.

        Yt=self.project(Yt)
        return Yt

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.vote = Vote()

    def forward(self, *input):
        matcher = Matcher()
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        corr_embedding = torch.matmul(tgt_embedding, scores.transpose(2, 1).contiguous())
        both_embedding = torch.cat((src_embedding, corr_embedding), dim=1)
        delta_corr = self.vote(both_embedding)

        src_corr = src_corr + delta_corr

        # src_centered = src - src.mean(dim=2, keepdim=True)
        # src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        src, src_corr = matcher.SC2_PCR(src.transpose(2,1), src_corr.transpose(2,1))

        # Rcolor = [1, 0, 0]
        # Gcolor = [0, 1, 0]
        # Bcolor = [0, 0, 0]
        # src_pcd = o3d.geometry.PointCloud()
        # src_pcd.points = o3d.utility.Vector3dVector(src.detach().cpu().squeeze(0).T.numpy())
        # src_pcd.paint_uniform_color(Gcolor)
        # VR_pcd = o3d.geometry.PointCloud()
        # VR_pcd.points = o3d.utility.Vector3dVector(src_corr.detach().cpu().squeeze(0).T.numpy())
        # VR_pcd.paint_uniform_color(Bcolor)
        # target_pcd = o3d.geometry.PointCloud()
        # target_pcd.points = o3d.utility.Vector3dVector(tgt.detach().cpu().squeeze(0).T.numpy())
        # target_pcd.paint_uniform_color(Rcolor)
        # o3d.visualization.draw_geometries([VR_pcd])

        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3), src, src_corr, scores,

class Vote(nn.Module):
    def __init__(self, out_dims=3, in_dims=1024):
        super(Vote, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 16, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(16, out_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(out_dims) # B,1024,n -> B,3,n

    def forward(self, x):
        x1 = F.leaky_relu((self.bn1(self.conv1(x))))
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)))
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)))
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)))
        return x5


class Matcher():
    def __init__(self,
                 inlier_threshold=0.0990,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=1024,
                 k1=30,
                 k2=20,
                 select_scene=None,
                 ):
        super(Matcher, self).__init__()
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.d_thre = d_thre
        self.num_iterations = num_iterations  # maximum iteration of power iteration algorithm
        self.ratio = ratio  # the maximum ratio of seeds.
        self.max_points = max_points
        self.nms_radius = nms_radius
        self.k1 = k1
        self.k2 = k2

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()

        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]

        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx

    def cal_seed_trans(self, seeds, SC2_measure, src_keypts, tgt_keypts):
        """
        Calculate the transformation for each seeding correspondences.
        Input:
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - SC2_measure: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
        Output: leading eigenvector
            - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
        """
        bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]
        k1 = self.k1
        k2 = self.k2

        if k1 > num_channels:
            k1 = 4
            k2 = 4

        #################################
        # The first stage consensus set sampling
        # Finding the k1 nearest neighbors around each seed
        #################################
        sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
        knn_idx = sorted_score[:, :, 0: k1]
        sorted_value, _ = torch.sort(SC2_measure, dim=2, descending=True)
        idx_tmp = knn_idx.contiguous().view([bs, -1])
        idx_tmp = idx_tmp[:, :, None]
        idx_tmp = idx_tmp.expand(-1, -1, 3)

        #################################
        # construct the local SC2 measure of each consensus subset obtained in the first stage.
        #################################
        src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_SC_measure = (cross_dist < self.d_thre).float()
        local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

        #################################
        # perform second stage consensus set sampling
        #################################
        sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
        knn_idx_fine = sorted_score[:, :, :, 0: k2]

        #################################
        # construct the soft SC2 matrix of the consensus set
        #################################
        num = knn_idx_fine.shape[1]
        knn_idx_fine = knn_idx_fine.contiguous().view([bs, num, -1])[:, :, :, None]
        knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
        src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])  # [bs, num_seeds, k, 3]
        tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])

        src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_measure = (cross_dist < self.d_thre * 2).float()
        local_SC2_measure = torch.matmul(local_hard_measure, local_hard_measure) / k2
        local_SC_measure = torch.clamp(1 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        # local_SC2_measure = local_SC_measure * local_SC2_measure
        local_SC2_measure = local_SC_measure
        local_SC2_measure = local_SC2_measure.view([-1, k2, k2])

        #################################
        # Power iteratation to get the inlier probability
        #################################
        local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(local_SC2_measure, method='power')
        total_weight = total_weight.view([bs, -1, k2])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k2])
        src_knn = src_knn_fine
        tgt_knn = tgt_knn_fine
        src_knn, tgt_knn = src_knn.view([-1, k2, 3]), tgt_knn.view([-1, k2, 3])

        #################################
        # compute the rigid transformation for each seed by the weighted SVD
        #################################
        seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        seedwise_fitness = torch.sum((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, num_seeds]
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        best_guess_ratio = seedwise_fitness[0, batch_best_guess]
        final_trans = seedwise_trans.gather(dim=1,
                                            index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

        return final_trans

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):
        """
        Calculate the confidence of the spectral matching solution based on spectral analysis.
        Input:
            - M:          [bs, num_corr, num_corr] the compatibility matrix
            - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
        Output:
            - confidence
        """
        if method == 'eig_value':
            # max eigenvalue as the confidence (Rayleigh quotient)
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                    leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            # max eigenvalue / second max eigenvalue as the confidence
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                    leading_eig[:, None, :] @ leading_eig[:, :, None])
            # compute the second largest eigen-value
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-6)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = (second_eig[:, None, :] @ B @ second_eig[:, :, None]) / (
                    second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            # max xMx as the confidence (x is the binary solution)
            # rank = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(M.shape[1]*self.ratio)]
            # binary_sol = torch.zeros_like(leading_eig)
            # binary_sol[0, rank[0]] = 1
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, it_num, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.05] * it_num
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)

            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num

        src_keypts = src_keypts[:, pred_inlier, :]
        tgt_keypts = tgt_keypts[:, pred_inlier, :]

        return src_keypts, tgt_keypts

    def match_pair(self, src_keypts, tgt_keypts, src_features, tgt_features):
        N_src = src_features.shape[1]
        N_tgt = tgt_features.shape[1]
        # use all point or sample points.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[:, src_sel_ind, :]
        tgt_desc = tgt_features[:, tgt_sel_ind, :]
        src_keypts = src_keypts[:, src_sel_ind, :]
        tgt_keypts = tgt_keypts[:, tgt_sel_ind, :]

        # match points in feature space.
        distance = torch.sqrt(2 - 2 * (src_desc[0] @ tgt_desc[0].T) + 1e-6)
        distance = distance.unsqueeze(0)
        source_idx = torch.argmin(distance[0], dim=1)
        corr = torch.cat([torch.arange(source_idx.shape[0])[:, None].cuda(), source_idx[:, None]], dim=-1)

        # generate correspondences
        src_keypts_corr = src_keypts[:, corr[:, 0]]
        tgt_keypts_corr = tgt_keypts[:, corr[:, 1]]

        return src_keypts_corr, tgt_keypts_corr

    def SC2_PCR(self, src_keypts, tgt_keypts):
        """
        Input:
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix.
            - pred_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
        """
        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]

        #################################
        # downsample points
        #################################
        if num_corr > self.max_points:
            src_keypts = src_keypts[:, :self.max_points, :]
            tgt_keypts = tgt_keypts[:, :self.max_points, :]
            num_corr = self.max_points

        #################################
        # compute cross dist
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(src_dist - target_dist)

        #################################
        # compute first order measure
        #################################
        SC_dist_thre = self.d_thre
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < SC_dist_thre).float()

        #################################
        # select reliable seed correspondences
        #################################
        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))

        #################################
        # compute second order measure
        #################################
        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                                      index=seeds[:, :, None].expand(-1, -1, num_corr))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                                                  index=seeds[:, :, None].expand(-1, -1, num_corr))
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure

        #################################
        # compute the seed-wise transformations and select the best one
        #################################
        final_trans = self.cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts)

        #################################
        # refine the result by recomputing the transformation over the whole set
        #################################
        src_corr, tgt_corr = self.post_refinement(final_trans, src_keypts, tgt_keypts, 20)

        return src_corr.transpose(2,1), tgt_corr.transpose(2,1)

def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans
