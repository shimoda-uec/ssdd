import datetime
import math
import os
import random
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import imutils
import utils
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
import time
from PIL import Image
from base_class import BaseModel, SegBaseModel, SSDDBaseModel, PascalDataset
from network import SegmentationPsa, PredictDiff, PredictDiffHead
import math

############################################################
#  dataset
############################################################

class SSDDValData(PascalDataset):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.joint_transform_list=[
                                imutils.Rescale(self.config.INP_SHAPE),
                                None,
                                None,
                            ]
        self.img_transform_list=[
                                np.asarray,
                                imutils.Normalize(mean = self.mean, std = self.std),
                                imutils.HWC_to_CHW
                            ]
    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        # Load image and mask
        impath= self.config.VOC_ROOT+'/JPEGImages/'
        imn=impath+image_id+'.jpg'
        img = Image.open(imn).convert("RGB")
        img = self.img_label_resize([img])[0]
        images = torch.from_numpy(img)
        return images, image_index

    def __len__(self):
        return self.image_ids.shape[0]

  
############################################################
#  Model Class
############################################################

class SegModel(SegBaseModel):
    def __init__(self, config):
        super(SegModel, self).__init__(config)
        in_channel=4096
        self.seg_main = SegmentationPsa(config, in_channel=in_channel, middle_channel=512, num_classes=21)

    def forward(self, inputs):
        x = inputs
        [x1, x2, x3, x4, x5] = self.encoder(x)
        seg, seg_head = self.seg_main(x5)
        return seg

class Evaluator():
    def __init__(self, config, model):
        super(Evaluator, self).__init__()
        self.config = config
        self.model=model

    def eval_model(self, val_dataset):
        self.val_set = SSDDValData(val_dataset, self.config)
        val_generator = torch.utils.data.DataLoader(self.val_set, batch_size=self.config.BATCH, shuffle=False, num_workers=torch.cuda.device_count()*2, pin_memory=True)
        self.model.eval()
        self.eval(val_generator)

    def get_segmentation(self, img):
        segs = self.get_ms_segout(img)
        fimg = img[:,:,:,torch.arange(img.shape[3]-1,-1,-1)]
        fsegs = self.get_ms_segout(fimg)
        seg_all = torch.zeros(1,segs[0].shape[1],segs[0].shape[2],segs[0].shape[3])
        for i in range(len(segs)):
            seg_all += segs[i]
        for i in range(len(segs)):
            seg_all += fsegs[i][:,:,:,torch.arange(fsegs[i].shape[3]-1,-1,-1)]
        return seg_all

    def get_ms_segout(self, img):
        scales = [1/2, 3/4, 1, 5/4, 3/2]
        segs = []
        for i in range(len(scales)):
          scale=scales[i]
          simg = F.interpolate(img, (int(img.shape[2]*scale),int(img.shape[3]*scale)), mode='bilinear')
          seg = self.model(simg)
          seg = F.softmax(seg,dim=1)
          seg = F.interpolate(seg, (int(img.shape[2]),int(img.shape[3])), mode='bilinear')
          seg = seg.data.cpu()
          segs.append(seg)
          torch.cuda.empty_cache()
        return segs

    def eval(self, datagenerator):
        end = time.time()
        cnt=0
        for inputs in datagenerator:
            print(cnt)
            data_time = time.time()
            start=time.time()
            images, imgindex = inputs
            images = Variable(images).cuda()
            segs=[]
            with torch.no_grad():
                for i in range(len(images)):
                    # segmentation
                    seg=self.get_segmentation(images[i:i+1])
                    # crf
                    image_id = self.val_set.image_ids[imgindex[i]]
                    impath=self.config.VOC_ROOT+'/JPEGImages/'
                    imn=impath+image_id+'.jpg'
                    img_org = np.asarray(Image.open(imn))
                    seg=F.interpolate(seg,(img_org.shape[0],img_org.shape[1]),mode='bilinear')
                    prob=F.softmax(seg,dim=1)[0].data.cpu().numpy()
                    seg_mask = np.argmax(prob,0)
                    seg_crf_map = imutils.crf_inference(img_org, prob, labels=prob.shape[0], t=10)
                    seg_crf_mask = np.argmax(seg_crf_map,axis=0)
                    # save results
                    cnt+=1
                    saven = os.path.join(self.savedir, 'seg_val_'+self.saveid+'_'+str(cnt)+'.png')
                    utils.mask2png(saven, seg_mask)
                    saven = os.path.join(self.savedir, 'seg_val_'+self.saveid+'_'+str(cnt)+'.txt')
                    np.savetxt(saven, seg_mask)
                    saven = os.path.join(self.savedir, 'seg_val_crf_'+self.saveid+'_'+str(cnt)+'.png')
                    utils.mask2png(saven, seg_crf_mask)
                    saven = os.path.join(self.savedir, 'seg_val_crf_'+self.saveid+'_'+str(cnt)+'.txt')
                    np.savetxt(saven, seg_crf_mask)



    def set_log_dir(self, phase, saveid, model_path=None):
            self.phase = phase
            self.saveid = saveid
            self.savedir = 'validation/'+self.saveid
            print("save the results to "+self.savedir)
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)

def val(config, weight_file=None):
    model = SegModel(config=config)
    return model
