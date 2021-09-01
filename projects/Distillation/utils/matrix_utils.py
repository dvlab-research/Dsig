import torch
from torch import nn
import torch.nn.functional as F


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def generate_correlation_matrix(feat, simf="instance_sim"):
    """
    :param feat:
    :param similarity_metric:
    :return:
    """
    correlation_matrix_list = []
    for feat_per_image in feat:
        # [M, C]
        if simf == "instance_sim":
            num_instances = feat_per_image.size(0)
            # import pdb;pdb.set_trace()
            # feat_per_image_row = feat_per_image.unsqueeze(2).expand(-1, -1, num_instances)
            # feat_per_image_col = feat_per_image.unsqueeze(2).expand(-1, -1, num_instances).transpose(0, 2)
            # sim = F.cosine_similarity(feat_per_image_row, feat_per_image_col, dim=1)

            feat_per_image_normalized = F.normalize(feat_per_image)
            sim = torch.mm(feat_per_image_normalized, feat_per_image_normalized.T)

        if simf == "channel_sim":
            num_instances = feat_per_image.size(0)
            feat_per_image = feat_per_image.view(num_instances, 256, -1)
            # [N, 256, 49]
            feat_per_image_row = feat_per_image.unsqueeze(3).expand(-1, -1, -1, num_instances)
            feat_per_image_col = feat_per_image.unsqueeze(3).expand(-1, -1, -1, num_instances).transpose(0, 3)
            # [N, 256, N]
            sim = F.cosine_similarity(feat_per_image_row, feat_per_image_col, dim=2)
            sim = sim.mean(dim=1)

        if feat_per_image.shape[0] == 0:
            sim = feat_per_image.sum() * 0.
        correlation_matrix_list.append(sim)
    return correlation_matrix_list


def channel_wise_cosine_sim(feat_i, feat_j):
    feat_i = feat_i.view(-1, 7, 7).flatten(1)
    feat_j = feat_j.view(-1, 7, 7).flatten(1)

    sim = F.cosine_similarity(feat_i, feat_j, dim=1)
    sim = sim.mean()
    return sim


def corr_mat_mse_loss(matlist_t, matlist_s, reduction='mean'):
    loss = 0.
    for mat_t, mat_s in zip(matlist_t, matlist_s):
        loss += 0.5 * F.mse_loss(mat_t, mat_s)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError('must specify reduction as none or mean.')


def split_features_per_image(feats_batch, nums_per_image):
    """
    :param feats_batch:
    :param nums_per_image:
    :return:
    """
    return torch.split(feats_batch, nums_per_image)


def fuse_bg_features(feats):
    """
    :param feats:
    :return:
    """
    fused_feats = []
    for feat in feats:
        feat = feat.flatten(1)
        feat = feat.mean(dim=0)
        fused_feats.append(feat)
    return fused_feats


def cat_fg_bg_features(feats_fg, feats_bg):
    """
    :param feats_fg:
    :param feats_bg:
    :return:
    """
    cated_feats = []
    for feat_fg, feat_bg in zip(feats_fg, feats_bg):
        cated_feat = torch.cat([feat_fg.flatten(1), feat_bg.flatten(1)], dim=0)
        cated_feats.append(cated_feat)
    return cated_feats


def select_topk_features_as_fg(feats, idx_list):
    sel_feats = []
    for feat, idx in zip(feats, idx_list):
        sel_feat = feat[idx]
        if len(sel_feat.shape) == 3:
            sel_feat = sel_feat.unsqueeze(0)
        sel_feats.append(sel_feat)
    return sel_feats

