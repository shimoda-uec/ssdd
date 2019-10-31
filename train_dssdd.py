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
from torchvision import transforms
import imutils
import utils
from base_class import BaseModel, SegBaseModel, SSDDBaseModel, PascalDataset
import ssdd_function as ssddF
import time
from PIL import Image
from network import SegmentationPsa, PredictDiff, PredictDiffHead
import math
import cv2
cv2.setNumThreads(0)

############################################################
#  dataset
############################################################

class DssddData(PascalDataset):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.label_dic = dataset.label_dic
        self.joint_transform_list=[
                                None,
                                imutils.RandomHorizontalFlip(),
                                imutils.RandomResizeLong(512, 768),
                                imutils.RandomCrop(448),
                                None,
                     ]
    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        # Load image and mask
        impath=''
        imn=impath+image_id+'.jpg'
        img = Image.open(imn).convert("RGB")
        gt_class_mlabel = torch.from_numpy(self.label_dic[image_id])
        gt_class_mlabel_bg = torch.from_numpy(np.concatenate(([1],self.label_dic[image_id])))
        psan = 'out_aff/'+image_id+'_psa.npy'
        psa=np.load(psan).transpose(1,2,0)
        psan = 'out_aff/'+image_id+'_psa_crf.npy'
        psa_crf=np.load(psan).transpose(1,2,0)
        h=psa.shape[0]
        w=psa.shape[1]
        saven = 'dd/dd0_'+str(image_index)+'.npz'
        dd0=np.load(saven)['arr_0'].transpose(1,2,0)
        dd0=np.reshape(cv2.resize(dd0,(w,h)),(h,w,1))
        saven = 'dd/dd1_'+str(image_index)+'.npz'
        dd1=np.load(saven)['arr_0'].transpose(1,2,0)
        dd1=np.reshape(cv2.resize(dd1,(w,h)),(h,w,1))
        img_norm, img_org, psa, psa_crf, dp0, dp1 = self.img_label_resize([img, np.array(img), psa, psa_crf, dd0, dd1])
        img_org = cv2.resize(img_org,self.config.OUT_SHAPE)
        dd0 = cv2.resize(dd0,self.config.OUT_SHAPE)
        dd1 = cv2.resize(dd1,self.config.OUT_SHAPE)
        psa = cv2.resize(psa,self.config.OUT_SHAPE)
        psa_crf = cv2.resize(psa_crf,self.config.OUT_SHAPE)
        psa=self.get_prob_label(psa, gt_class_mlabel_bg).transpose(2,0,1)
        psa_crf=self.get_prob_label(psa_crf, gt_class_mlabel_bg).transpose(2,0,1)
        psa_mask = np.argmax(psa,0)
        psa_crf_mask = np.argmax(psa_crf,0)
        dd0 = torch.from_numpy(dd0).unsqueeze(0)
        dd1 = torch.from_numpy(dd1).unsqueeze(0)
        psa_mask = torch.from_numpy(psa_mask).unsqueeze(0)
        psa_crf_mask = torch.from_numpy(psa_crf_mask).unsqueeze(0)
        ignore_flags=torch.from_numpy(ssddF.get_ignore_flags(psa_mask, psa_crf_mask, [gt_class_mlabel])).float()
        (_, _, _, seed_mask) = ssddF.get_dd_mask(dd0, dd1, psa_mask, psa_crf_mask, ignore_flags, dd_bias=0.1, bg_bias=0.1)
        return img_norm, img_org, gt_class_mlabel, gt_class_mlabel_bg, seed_mask[0]
    def __len__(self):
        return self.image_ids.shape[0]

############################################################
#  Models
############################################################
class SegModel(SegBaseModel):
    def __init__(self, config):
        super(SegModel, self).__init__(config)
        self.config = config
        in_channel=4096
        self.seg_main = SegmentationPsa(config,num_classes=21, in_channel=in_channel, middle_channel=512, scale=2)
        self.seg_sub = SegmentationPsa(config,num_classes=21, in_channel=in_channel, middle_channel=512, scale=2)
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        self.apply(set_bn_fix)
    def forward(self, inputs):
        x, img_org, gt_class_mlabel = inputs
        feats = self.encoder(x)
        [x1,x2,x3,x4,x5] = feats
        seg_outs_main = self.get_seg(self.seg_main, x5, gt_class_mlabel)
        seg_outs_sub = self.get_seg(self.seg_sub, x5, gt_class_mlabel)
        seg_crf, seg_crf_mask = self.get_crf(img_org, seg_outs_main[0], gt_class_mlabel)
        return seg_outs_main, seg_outs_sub, seg_crf_mask, feats

class SSDDModel(SSDDBaseModel):
    def __init__(self, config):
        super(SSDDModel, self).__init__(config)
        self.dd_head0 = PredictDiffHead(config, in_channel=512, in_channel2=128)
        self.dd_head1 = PredictDiffHead(config, in_channel=512, in_channel2=128)
        self.dd0 = PredictDiff(config, in_channel=256, in_channel2=128)
        self.dd1 = PredictDiff(config, in_channel=256, in_channel2=128)
    def forward(self, inputs):
        (seg_outs_main, seg_outs_sub, seg_crf_mask, feats), seed_mask, gt_class_mlabel = inputs
        [x1,x2,x3,x4,x5] = feats
        x1=F.avg_pool2d(x1, 2, 2)
        # first step
        seg_main, seg_prob_main, seg_mask_main, seg_head_main = seg_outs_main
        ignore_flags0=torch.from_numpy(ssddF.get_ignore_flags(seg_mask_main, seg_crf_mask, gt_class_mlabel)).cuda().float()
        dd_head0 = self.dd_head0((seg_head_main.detach(), x1.detach()))
        dd00 = ssddF.get_dd(self.dd0, dd_head0, seg_mask_main)
        dd01 = ssddF.get_dd(self.dd0, dd_head0, seg_crf_mask)
        dd_outs0 = ssddF.get_dd_mask(dd00, dd01, seg_mask_main, seg_crf_mask, ignore_flags0, dd_bias=0.4, bg_bias=0)
        (dd01, dd10, ignore_flags0, refine_mask0)=dd_outs0
        # second step
        seg_sub, seg_prob_sub, seg_mask_sub, seg_head_sub = seg_outs_sub
        dd_head1 = self.dd_head1((seg_head_sub.detach(), x1.detach()))
        dd10 = ssddF.get_dd(self.dd1, dd_head1, seed_mask)
        dd11 = ssddF.get_dd(self.dd1, dd_head1, refine_mask0)
        ignore_flags1 = torch.from_numpy(ssddF.get_ignore_flags(seed_mask, refine_mask0, gt_class_mlabel)).cuda().float()
        dd_outs1 = ssddF.get_dd_mask(dd10, dd11, seed_mask, refine_mask0, ignore_flags1, dd_bias=0.4, bg_bias=0)
        return dd_outs0, dd_outs1

############################################################
#  Trainer
############################################################
class Trainer():
    def __init__(self, config, model_dir, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.epoch = 0
        self.layer_regex = {
            "lr1": r"(encoder.*)",
            "lr10": r"(seg_main.*)|(seg_sub.*)",
            "dd": r"(dd0.*)|(dd1.*)|(dd_head0.*)|(dd_head1.*)",
        }
        lr_1x = self.layer_regex["lr1"]
        lr_10x = self.layer_regex["lr10"]
        dd = self.layer_regex['dd']
        seg_model=model[0].cuda()
        ssdd_model=model[1].cuda()
        self.param_lr_1x = [param for name, param in seg_model.named_parameters() if bool(re.fullmatch(lr_1x, name)) and not 'bn' in name]
        self.param_lr_10x = [param for name, param in seg_model.named_parameters() if bool(re.fullmatch(lr_10x, name)) and not 'bn' in name]
        self.param_dd = [param for name, param in ssdd_model.named_parameters() if bool(re.fullmatch(dd, name)) and not 'bn' in name]
        lr=1e-3
        self.seg_model=nn.DataParallel(seg_model)
        self.ssdd_model=nn.DataParallel(ssdd_model)
    def train_model(self, train_dataset, layers):
        epochs=self.config.EPOCHS
        # Data generators
        self.train_set = DssddData(train_dataset, self.config)
        train_generator = torch.utils.data.DataLoader(self.train_set, batch_size=self.config.BATCH, shuffle=True, num_workers=8, pin_memory=True)
        self.config.LR_RAMPDOWN_EPOCHS=int(epochs*1.2)
        self.seg_model.train()
        self.ssdd_model.train()
        for epoch in range(0, epochs):
            print("Epoch {}/{}.".format(epoch,epochs))
            # Training
            self.train_epoch(train_generator, epoch)
            # Save model
            if (epoch % 2 ==0) & (epoch>0):
                torch.save(self.model_seg.state_dict(), self.checkpoint_path_seg.format(epoch))
                torch.save(self.model_ssdd.state_dict(), self.checkpoint_path_ssdd.format(epoch))
            torch.cuda.empty_cache()
    def train_epoch(self, datagenerator, epoch):
        learning_rate=self.config.LEARNING_RATE
        self.cnt=0
        self.steps = len(datagenerator)
        self.step=0
        self.epoch=epoch
        end=time.time()
        for inputs in datagenerator:
            self.train_step(inputs, end)
            end=time.time()
            self.step += 1
    def train_step(self, inputs, end):
        start = time.time()
        lr=utils.adjust_learning_rate(self.config.LEARNING_RATE, self.epoch, self.config.LR_RAMPDOWN_EPOCHS, self.step, self.steps)
        self.optimizer = torch.optim.SGD([
            {'params': self.param_lr_1x,'lr': lr*1, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': self.param_lr_10x,'lr': lr*10, 'weight_decay': self.config.WEIGHT_DECAY},
        ], lr=lr, momentum=self.config.LEARNING_MOMENTUM, weight_decay= self.config.WEIGHT_DECAY)
        self.optimizer_dd = torch.optim.SGD([
            {'params': self.param_dd,'lr': lr*10, 'weight_decay': self.config.WEIGHT_DECAY},
        ], lr=lr, momentum=self.config.LEARNING_MOMENTUM, weight_decay= self.config.WEIGHT_DECAY)
        img_norm, img_org, gt_class_mlabels, gt_class_mlabels_bg, seed_mask = inputs
        img_norm = Variable(img_norm).cuda().float()
        img_org = Variable(img_org).cuda().float()
        seed_mask = Variable(seed_mask).cuda().long()
        gt_class_mlabels = Variable(gt_class_mlabels).cuda().float()
        gt_class_mlabels_bg = Variable(gt_class_mlabels_bg).cuda().float()
        seg_outs = self.seg_model((img_norm, img_org, gt_class_mlabels_bg))
        dd_outs = self.ssdd_model((seg_outs, seed_mask, gt_class_mlabels))
        loss_seg, loss_dd = self.compute_loss(seg_outs, dd_outs, inputs)
        forward_time=time.time()
        self.optimizer.zero_grad()
        loss_seg.backward()
        self.optimizer.step()
        forward_time=time.time()
        self.optimizer_dd.zero_grad()
        loss_dd.backward()
        self.optimizer_dd.step()
        forward_time=time.time()
        if (self.step%10==0):
            prefix="{}/{}/{}/{}".format(self.epoch, self.cnt, self.step + 1, self.steps)
            suffix="forward_time: {:.3f} time: {:.3f} data {:.3f} seg: {:.3f}".format(
                forward_time-start, (time.time()-start),(start-end),loss_seg.item())
            print('\r%s %s' % (prefix, suffix), end = '\n')

    def compute_loss(self, seg_outs, dd_outs, inputs):
        seg_outs_main, seg_outs_sub, seg_crf_mask, feats = seg_outs
        seg_main, seg_prob_main, seg_mask_main, _ = seg_outs_main
        seg_sub, seg_prob_sub, seg_mask_sub, _ = seg_outs_sub
        dd_outs0, dd_outs1 = dd_outs
        images, img_org, gt_class_mlabels, gt_class_mlabels_bg, seed_mask = inputs
        seed_mask = Variable(seed_mask).cuda().long()
        (dd00, dd01, ignore_flags0, refine_mask0) = dd_outs0
        (dd10, dd11, ignore_flags1, refine_mask1) = dd_outs1
        loss_seg_main = F.cross_entropy(seg_main, refine_mask1, ignore_index=255)
        loss_seg_sub = 0.5*F.cross_entropy(seg_sub, seed_mask, ignore_index=255) + 0.5*F.cross_entropy(seg_sub, refine_mask1, ignore_index=255)
        loss_seg = loss_seg_main + loss_seg_sub
        seg_crf_diff = seg_mask_main != seg_crf_mask
        loss_dd00 = ssddF.get_ddloss(dd00, seg_crf_diff, ignore_flags0)
        loss_dd01 = ssddF.get_ddloss(dd01, seg_crf_diff, ignore_flags0)
        loss_dd10 = ssddF.compute_sig_mask_loss(dd10, seed_mask != seg_mask_sub)
        loss_dd11 = ssddF.compute_sig_mask_loss(dd11, refine_mask1 != seg_mask_sub)
        loss_dd = (loss_dd00 + loss_dd01 + loss_dd10 + loss_dd11)/4
        return loss_seg, loss_dd

    def set_log_dir(self, runner_name, phase, saveid, model_path=None):
            # Set date and epoch counter as if starting a new model
            self.epoch = 0
            self.phase = phase
            self.saveid = saveid
            self.log_dir = os.path.join(self.model_dir, "{}_{}_{}".format(runner_name, phase, saveid))
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path_seg = os.path.join(self.log_dir, "seg_*epoch*.pth".format())
            self.checkpoint_path_seg = self.checkpoint_path_seg.replace("*epoch*", "{:04d}")
            self.checkpoint_path_ssdd = os.path.join(self.log_dir, "ssdd_*epoch*.pth".format())
            self.checkpoint_path_ssdd = self.checkpoint_path_ssdd.replace("*epoch*", "{:04d}")

def models(config, weight_file=None):
    seg_model = SegModel(config=config)
    seg_model.initialize_weights()
    seg_model.load_resnet38_weights(weight_file)
    ssdd_model = SSDDModel(config=config)
    ssdd_model.initialize_weights()
    return (seg_model, ssdd_model)
