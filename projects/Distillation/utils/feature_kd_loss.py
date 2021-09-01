import torch
from torch import nn
import torch.nn.functional as F

similarity_func_dict = {
    "cosine_sim_d0": lambda f1, f2: cosine_similarity(f1, f2, reserve_dim=0),
    "cosine_sim_d1": lambda f1, f2: cosine_similarity(f1, f2, reserve_dim=1),
    "kl_sim_d0": lambda f1, f2: kl_similarity(f1, f2, reserve_dim=0),
    "kl_sim_d1_sum": lambda f1, f2: kl_similarity(f1, f2, reserve_dim=1, reduction='sum'),
    "kl_sim_d1_mean": lambda f1, f2: kl_similarity(f1, f2, reserve_dim=1, reduction='mean'),
    "kl_sim": lambda f1, f2: kl_similarity(f1, f2, reserve_dim=-1),
    "l2_sim": lambda f1, f2: l2_similarity(f1, f2)
}

def mse_loss_withmask(fs, ft, fmask=None):
    lmap = F.mse_loss(fs, ft, reduction='none')
    if fmask == None:
        return lmap.mean()
    lmap = lmap * fmask.unsqueeze(1)
    lmap = lmap.flatten(1).sum(1)
    fmask = fmask.flatten(1).sum(1)
    loss = lmap / fmask / 256 
    loss = loss.mean()
    return loss

def l2_similarity(feat, feat_ori, reserve_dim=0):
    # L2 distance
    if len(feat.size()) == 2:
        N, C = feat.size()
        sim = F.mse_loss(
            feat.view(N, -1), feat_ori.view(N, -1), reduction='none'
        ).mean(dim=1)

        return sim
    else:
        raise ValueError()

def cosine_similarity(feat, feat_ori, reserve_dim=0):
    # import pdb;pdb.set_trace()
    if len(feat.size()) == 4:
        N, C, H, W = feat.size()
    elif len(feat.size()) == 3:
        N, C, FC = feat.size()
    elif len(feat.size()) == 2:
        N, C = feat.size()
    else:
        raise ValueError()
    if reserve_dim == 0:
        sim = F.cosine_similarity(
            feat.contiguous().view(N, -1),
            feat_ori.contiguous().view(N, -1),
            dim=1
        )
    elif reserve_dim == 1:
        sim = F.cosine_similarity(
            feat.contiguous().view(N*C, -1),
            feat_ori.contiguous().view(N*C, -1),
            dim=1
        ).view(N, C)
    else:
        raise ValueError("Reserve dim must be 0 or 1 in cosine similarity")
    return sim

def kl_similarity(feat, feat_ori, reserve_dim=-1, reduction=None):
    # loss mags are not correlated with grad mags
    N, C, H, W = feat.size()

    if reserve_dim == -1:
        sim = F.kl_div(
            F.log_softmax(feat.view(N, -1), dim=1),
            F.softmax(feat_ori.view(N, -1), dim=1),
            reduction='none'
        ).view(N, C, H, W)
    elif reserve_dim == 0:
        sim = F.kl_div(
            F.log_softmax(feat.view(N, -1), dim=1),
            F.softmax(feat_ori.view(N, -1), dim=1),
            reduction='none'
        ).sum(dim=1).view(N, 1)
    elif reserve_dim == 1:
        if reduction == 'sum':
            sim = F.kl_div(
                F.log_softmax(feat.view(N * C, -1), dim=1),
                F.softmax(feat_ori.view(N * C, -1), dim=1),
                reduction='none'
            ).sum(dim=1).view(N, C)
        elif reduction == 'mean':
            sim = F.kl_div(
                F.log_softmax(feat.view(N * C, -1), dim=1),
                F.softmax(feat_ori.view(N * C, -1), dim=1),
                reduction='none'
            ).mean(dim=1).view(N, C)
        else:
            raise ValueError("must specify reduction as sum or mean")
    else:
        raise ValueError("Reserve dim must be 0 or 1 in cosine similarity")
    return sim

def kl_similarity_loss(sim_s, sim_t):
    """
    :param sim_s: KL similarity [N, C, H, W]
    :param sim_t: KL similarity [N, C, H, W]
    :return: mse loss
    """
    N = sim_s.size(0)
    sim_s = sim_s.view(N, -1)
    sim_t = sim_t.view(N, -1)

    lmap = F.mse_loss(sim_s, sim_t, reduction='none')

    return lmap.sum()


def similarity_mse_loss(sim_s, sim_t):
    N = sim_s.size(0)
    sim_s = sim_s.view(N, -1)
    sim_t = sim_t.view(N, -1)

    lmap = F.mse_loss(sim_s, sim_t, reduction='none')

    return lmap.mean()

def similarity_l1_loss(sim_s, sim_t):
    N = sim_s.size(0)
    sim_s = sim_s.view(N, -1)
    sim_t = sim_t.view(N, -1)

    lmap = F.l1_loss(sim_s, sim_t, reduction='none')

    return lmap.mean()

def normalized_loss(losses, loss_weight=1.0, momentum=0.9):
    loss = losses["loss"]
    loss_bar = losses["loss_bar"]
    loss_aux = losses["loss_aux"]
    loss_aux_bar = losses["loss_aux_bar"]

    # update loss bar
    loss_bar = momentum * loss_bar + (1 - momentum) * loss.detach()
    loss_aux_bar = momentum * loss_aux_bar + (1 - momentum) * loss_aux.detach()

    loss = 1. / (1 + loss_weight) * loss
    loss_aux = loss_weight / (1 + loss_weight) * loss_bar / loss_aux_bar * loss_aux

    return loss, loss_aux, loss_bar, loss_aux_bar


