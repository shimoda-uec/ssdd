import torch
import torch.nn as nn
from arch_resnet38 import Resnet38
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import imutils
import os
import re

class BaseModel(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def load_resnet38_weights(self, filepath): 
        print(filepath, os.path.exists(filepath))
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            new_params = self.state_dict().copy()
            for i in new_params:
              i_parts = i.split('.')
            for i in state_dict:
              i_parts = i.split('.')
              if re.fullmatch('(fc8)', i_parts[0]):
                 pass
              else:
                 tmp=i_parts.copy()
                 tmp.insert(0,'encoder')
                 tmp='.'.join(tmp)
                 new_params[tmp] = state_dict[i]
            self.load_state_dict(new_params)
    def train(self, mode=True):
        super().train(mode)
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

class SegBaseModel(BaseModel):
    def __init__(self, config):
        super(SegBaseModel, self).__init__()
        self.config = config
        self.encoder=Resnet38()
    def get_crf(self, img_org, seg, gt_class_mlabel):
        img_org=img_org.data.cpu().numpy().astype(np.uint8)
        seg_crf=np.zeros((seg.shape[0],seg.shape[1],self.config.OUT_SHAPE[0],self.config.OUT_SHAPE[1]))
        for i in range(len(seg)):
            prob=[]
            for j in range(gt_class_mlabel.shape[1]):
                if gt_class_mlabel[i,j].item()==1:
                    prob.append(seg[i,j:j+1])
            prob=F.softmax(torch.cat(prob),dim=0).data.cpu().numpy()
            crf_map = imutils.crf_inference(img_org[i].copy(order='C'),prob,labels=prob.shape[0])
            cnt=0
            for j in range(gt_class_mlabel.shape[1]):
                if gt_class_mlabel[i,j].item()==1:
                    seg_crf[i][j]=crf_map[cnt]
                    cnt += 1
        seg_crf=torch.from_numpy(seg_crf).cuda().float()
        _, seg_crf_mask=torch.max(seg_crf,1)
        return seg_crf, seg_crf_mask

    def get_seg(self, segment_module, x5, gt_class_mlabel):
        seg, seg_head = segment_module(x5)
        seg_prob=F.softmax(seg,dim=1)
        gt_class_mlabel_maps = gt_class_mlabel.view(gt_class_mlabel.shape[0],gt_class_mlabel.shape[1],1,1).repeat(1,1,seg.shape[2],seg.shape[3])
        seg_prob=seg_prob*gt_class_mlabel_maps+gt_class_mlabel_maps*1e-4
        _,seg_mask = torch.max(seg_prob,1)
        return (seg, seg_prob, seg_mask, seg_head)

class SSDDBaseModel(BaseModel):
    def __init__(self, config):
        super(SSDDBaseModel, self).__init__()
        self.config = config

class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        self.joint_transform_list=[
                                None,
                                imutils.RandomHorizontalFlip(),
                                imutils.RandomResizeLong(512, 832),
                                imutils.RandomCrop(448),
                                None,
                     ]
        self.img_transform_list=[
                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                np.asarray,
                                None,
                                imutils.Normalize(mean = self.mean, std = self.std),
                                imutils.HWC_to_CHW
                    ]
    def img_label_resize(self, inputs):
        for joint_transform, img_transform in zip(self.joint_transform_list, self.img_transform_list):
            img_norm = inputs[0]
            if img_transform:
               img_norm = img_transform(img_norm)
               inputs[0]=img_norm
            if joint_transform:
               outputs = joint_transform(inputs)
               inputs=outputs
        return inputs
    def get_prob_label(self, prob, mlabel):
        # prob shape [HxWxC]
        # mlabel shape [C]
        prob_label=np.zeros((prob.shape[0],prob.shape[1],mlabel.shape[0]))
        cnt=0
        for i in range(0,mlabel.shape[0]):
            if mlabel[i]==1:
                prob_label[:,:,i]=prob[:,:,cnt]
                cnt+=1
        return prob_label
    def __len__(self):
        return self.image_ids.shape[0]

