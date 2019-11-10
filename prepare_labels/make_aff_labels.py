import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np
import cv2
import voc12.data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
from tool import imutils, pyutils
import time
from PIL import Image
import os
#voc12_root=os.environ['voc_root']
voc12_root="../voc_root"


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

def mask2png(mask,saven):
    palette = get_palette(256)
    mask=Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask.save(saven)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--logt", default=8, type=int)

    args = parser.parse_args()

    model = getattr(importlib.import_module("network.resnet38_aff"), 'Net')()

    model.load_state_dict(torch.load("res38_aff.pth"))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDataset(args.infer_list, voc12_root=voc12_root,
                                               transform=torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #save_dir=str(args.compatg)+"_"+str(args.compatb)+"_"+str(args.gxy)+"_"+str(args.bxy)+"_"+str(args.brgb)
    save_dir_cam = "results/out_cam"
    save_dir_aff = "results/out_aff"
    os.makedirs(save_dir_aff, exist_ok=True)
    save_dir_aff_crf = "results/out_aff_crf"
    os.makedirs(save_dir_aff_crf, exist_ok=True)

    for iter, (name, img, label) in enumerate(infer_data_loader):
        name = name[0]; label = label[0]
        img_path=voc12_root+'/JPEGImages/'+name+'.jpg'
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        print(iter)
        orig_shape = img.shape

        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        cam = np.load(os.path.join(save_dir_cam, name + '.npy')).item()
        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**args.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')
        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)
            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)#D sum(W)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)
            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
            cam_vec = cam_full_arr.view(21, -1)
            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
            cam_rw = cam_rw[:,:,:orig_shape[2], :orig_shape[3]]
            cam_rw = cam_rw.squeeze().data.cpu().numpy()

        aff_dict = {}
        aff_dict[0] = cam_rw[0]
        for i in range(20):
            if label[i] > 1e-5:
                aff_dict[i+1] = cam_rw[i+1]
        np.save(os.path.join(save_dir_aff, name + '.npy'), aff_dict)
        mask=np.argmax(cam_rw,axis=0)
        mask2png(mask, os.path.join(save_dir_aff, name + '.png'))

        v = np.array(list(aff_dict.values()))
        aff_crf = imutils.crf_inference(orig_img, v, labels=v.shape[0])
        aff_crf_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        cnt=0
        for k, v in aff_dict.items():
            aff_crf_full_arr[k] = aff_crf[cnt]
            cnt+=1
        np.save(os.path.join(save_dir_aff_crf, name + '.npy'), aff_crf)
        mask=np.argmax(aff_crf_full_arr,axis=0)
        mask2png(mask, os.path.join(save_dir_aff_crf, name + '.png'))
