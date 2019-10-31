import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
sigmoid = torch.nn.Sigmoid()
def compute_sig_mask_loss(logits, bin_mask):
    bin_mask=bin_mask.float()
    logits=sigmoid(logits).squeeze(1)
    loc0=bin_mask==0
    loc1=bin_mask==1
    logits0=logits[loc0]
    logits1=logits[loc1]
    bin_mask0=bin_mask[loc0]
    bin_mask1=bin_mask[loc1]
    loss0=F.binary_cross_entropy(logits0, bin_mask0)
    loss1=F.binary_cross_entropy(logits1, bin_mask1)
    return (loss0 + loss1)/2

def add_class_weights(pixel_weights, mask0, mask1, ignore_flags, bg_bias=0.00):
    for i in range(len(mask0)):
        pixel_weight = pixel_weights[i]
        pixel_weight -= (mask0[i]==(0)).float()*(bg_bias)
        pixel_weight += (mask1[i]==(0)).float()*(bg_bias)
        for j in range(1,ignore_flags.shape[1]):
            pixel_weight -= (mask0[i]==(j)).float()*(ignore_flags[i,j]*1.0)
            pixel_weight += (mask1[i]==(j)).float()*(ignore_flags[i,j]*1.0)
    return pixel_weights
def get_dd_mask(dd0, dd1, mask0, mask1, ignore_flags, dd_bias=0.15, bg_bias=0.05):
    dd0_prob = sigmoid(dd0)
    dd1_prob = sigmoid(dd1)
    w = dd0_prob-dd1_prob+dd_bias
    w = add_class_weights(w, mask0, mask1, ignore_flags, bg_bias=bg_bias)
    refine_mask=Variable(torch.zeros_like(mask0))+255
    bsc=((w.squeeze(1)>=0))
    bcs=bsc==0
    refine_mask[bsc]=mask1[bsc]
    refine_mask[bcs]=mask0[bcs]
    return (dd0, dd1, ignore_flags, refine_mask)
def get_dd(dd, dd_head, mask):
    binmask = get_binarymask(mask)
    dd_pred = dd((dd_head, binmask.detach()))
    return dd_pred

def get_ignore_flags(mask0, mask1, mlabel, th=0.5):
    ignore_flags=np.zeros((len(mask0),21,))
    for i in range(len(mlabel)):
        for j in range(len(mlabel[0])):
            if mlabel[i][j]==1:
                loc0=torch.sum(mask0[i]==(j+1)).item()
                loc1=torch.sum(mask1[i]==(j+1)).item()
                rate=loc1/max(loc0,1)
                if rate < th:
                    ignore_flags[i,j+1]=1
    return ignore_flags

def get_binarymask(masks, chn=21):
    # input [NxHxW]
    N,H,W=masks.shape
    bin_masks=torch.zeros(N,chn,H,W).cuda()
    for n in range(N):
      mask = masks[n]
      for c in range(chn):
        bin_mask = bin_masks[n,c]
        loc = mask==c
        locn=torch.sum(loc)
        if locn.sum()>0:
          bin_mask[loc]=1
    return bin_masks

def get_ddloss(dd, diff_mask, ignore_flags):
    loss_dd = Variable(torch.FloatTensor([0]),requires_grad=True).cuda()
    cnt=0
    for k in range(len(dd)):
        if torch.sum(ignore_flags[k,1:]).item()>0:
            continue
        cnt+=1
        loss_dd += compute_sig_mask_loss(dd[k:k+1], diff_mask[k:k+1])
    if cnt >0:
        loss_dd /= cnt
    return loss_dd
