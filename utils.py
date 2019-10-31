import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
def adjust_learning_rate(lr, epoch, lr_rampdown_epochs, step_in_epoch, total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
    lr *= cosine_rampdown(epoch, lr_rampdown_epochs)
    return lr

def get_labeled_tensor(tensor, class_label):
    labeled_tensor=[]
    for i in range(len(tensor)):
        for i in range(class_mlabel.shape[1]):
            if gt_class_mlabel[i,j].item()==1:
                tmp_prob.append(tensor[i:i+1,j:j+1])
        tmp_prob=torch.cat(tmp_prob)

def mask2png(saven, mask):
    palette = get_palette(256)
    mask=Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask.save(saven)

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
