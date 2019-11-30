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

class SssddData(PascalDataset):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.label_dic = dataset.label_dic
        self.joint_transform_list=[
                                None,
                                imutils.RandomHorizontalFlip(),
                                imutils.RandomResizeLong(448, 512),
                                imutils.RandomCrop(448),
                                None,
                     ]
    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        impath = self.config.VOC_ROOT+'/JPEGImages/'
        imn = impath+image_id+'.jpg'
        img = Image.open(imn).convert("RGB")
        gt_class_mlabel = torch.from_numpy(self.label_dic[image_id])
        gt_class_mlabel_bg = torch.from_numpy(np.concatenate(([1],self.label_dic[image_id])))
        psan = 'prepare_labels/results/out_aff/'+image_id+'.npy'
        psa=np.array(list(np.load(psan).item().values())).transpose(1,2,0)
        psan = 'prepare_labels/results/out_aff_crf/'+image_id+'.npy'
        psa_crf=np.load(psan).transpose(1,2,0)
        h=psa.shape[0]
        w=psa.shape[1]
        # resize inputs
        img_norm, img_org, psa, psa_crf = self.img_label_resize([img, np.array(img), psa, psa_crf])
        img_org = cv2.resize(img_org,self.config.OUT_SHAPE)
        psa = cv2.resize(psa,self.config.OUT_SHAPE)
        psa_crf = cv2.resize(psa_crf,self.config.OUT_SHAPE)
        psa=self.get_prob_label(psa, gt_class_mlabel_bg).transpose(2,0,1)
        psa_crf=self.get_prob_label(psa_crf, gt_class_mlabel_bg).transpose(2,0,1)
        psa_mask = np.argmax(psa,0)
        psa_crf_mask = np.argmax(psa_crf,0)
        return img_norm, img_org, gt_class_mlabel, gt_class_mlabel_bg, psa_mask, psa_crf_mask
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
        return seg_outs_main, feats

class SSDDModel(SSDDBaseModel):
    def __init__(self, config):
        super(SSDDModel, self).__init__(config)
        self.dd_head0 = PredictDiffHead(config, in_channel=512, in_channel2=128)
        self.dd0 = PredictDiff(config, in_channel=256, in_channel2=128)
    def forward(self, inputs):
        (seg_outs_main, feats), psa_mask, psa_crf_mask, gt_class_mlabel = inputs
        [x1,x2,x3,x4,x5] = feats
        x1=F.avg_pool2d(x1, 2, 2)
        # first step
        seg_main, seg_prob_main, seg_mask_main, seg_head_main = seg_outs_main
        ignore_flags0=torch.from_numpy(ssddF.get_ignore_flags(psa_mask, psa_crf_mask, gt_class_mlabel)).cuda().float()
        dd_head0 = self.dd_head0((seg_head_main.detach(), x1.detach()))
        dd00 = ssddF.get_dd(self.dd0, dd_head0, psa_mask)
        dd01 = ssddF.get_dd(self.dd0, dd_head0, psa_crf_mask)
        dd_outs0 = ssddF.get_dd_mask(dd00, dd01, psa_mask, psa_crf_mask, ignore_flags0, dd_bias=0.4, bg_bias=0)
        return dd_outs0

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
            "lr10": r"(seg_main.*)",
            "dd": r"(dd0.*)|(dd_head0.*)",
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
    def train_model(self, train_dataset):
        epochs=self.config.EPOCHS
        # Data generators
        self.train_set = SssddData(train_dataset, self.config)
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
                torch.save(self.seg_model.state_dict(), self.checkpoint_path_seg.format(epoch))
                torch.save(self.ssdd_model.state_dict(), self.checkpoint_path_ssdd.format(epoch))
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
        # adjust learning rate
        lr=utils.adjust_learning_rate(self.config.LEARNING_RATE, self.epoch, self.config.LR_RAMPDOWN_EPOCHS, self.step, self.steps)
        self.optimizer = torch.optim.SGD([
            {'params': self.param_lr_1x,'lr': lr*1, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': self.param_lr_10x,'lr': lr*10, 'weight_decay': self.config.WEIGHT_DECAY},
        ], lr=lr, momentum=self.config.LEARNING_MOMENTUM, weight_decay= self.config.WEIGHT_DECAY)
        self.optimizer_dd = torch.optim.SGD([
            {'params': self.param_dd,'lr': lr*10, 'weight_decay': self.config.WEIGHT_DECAY},
        ], lr=lr, momentum=self.config.LEARNING_MOMENTUM, weight_decay= self.config.WEIGHT_DECAY)
        # input items
        img_norm, img_org, gt_class_mlabel, gt_class_mlabel_bg, psa_mask, psa_crf_mask = inputs
        img_norm = Variable(img_norm).cuda().float()
        img_org = Variable(img_org).cuda().float()
        gt_class_mlabel = Variable(gt_class_mlabel).cuda().float()
        gt_class_mlabel_bg = Variable(gt_class_mlabel_bg).cuda().float()
        # forward
        seg_outs = self.seg_model((img_norm, img_org, gt_class_mlabel_bg))
        dd_outs = self.ssdd_model((seg_outs, psa_mask, psa_crf_mask, gt_class_mlabel))
        # get loss
        loss_seg, loss_dd = self.compute_loss(seg_outs, dd_outs, inputs)
        forward_time=time.time()
        # backward
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
        seg_outs_main, feats = seg_outs
        seg_main, seg_prob_main, seg_mask_main, _ = seg_outs_main
        dd_outs0 = dd_outs
        img_norm, img_org, gt_class_mlabel, gt_class_mlabel_bg, psa_mask, psa_crf_mask = inputs
        (dd00, dd01, ignore_flags0, refine_mask0) = dd_outs0
        psa_mask = Variable(psa_mask).cuda().long()
        psa_crf_mask = Variable(psa_crf_mask).cuda().long()
        # compute losses
        # segmentation loss
        loss_seg_main = F.cross_entropy(seg_main, psa_mask, ignore_index=255)
        loss_seg = loss_seg_main
        # difference detection loss
        psa_diff = psa_mask != psa_crf_mask
        loss_dd00 = ssddF.get_ddloss(dd00, psa_diff, ignore_flags0)
        loss_dd01 = ssddF.get_ddloss(dd01, psa_diff, ignore_flags0)
        loss_dd = (loss_dd00 + loss_dd01)/2
        # save temporary outputs
        if (self.step%30==0):
            sid='_'+self.phase+'_'+self.saveid+'_'+str(self.epoch)+'_'+str(self.cnt)
            img_org=img_org.data.cpu().numpy()[...,::-1]
            saven = self.log_dir_img + '/i'+sid+'.jpg'
            cv2.imwrite(saven,img_org[0])
            saven = self.log_dir_img + '/D'+sid+'.png'
            mask_png = utils.mask2png(saven, refine_mask0[0].squeeze().data.cpu().numpy())
            saven = self.log_dir_img + '/K'+sid+'.png'
            mask_png = utils.mask2png(saven, psa_mask[0].data.cpu().numpy().astype(np.float32))
            saven = self.log_dir_img + '/A'+sid+'.png'
            mask_png = utils.mask2png(saven, psa_crf_mask[0].squeeze().data.cpu().numpy())

            saven = self.log_dir_img + 'da'+sid+'.png'
            tmp=F.sigmoid(dd00)[0].squeeze().data.cpu().numpy()
            cv2.imwrite(saven,tmp*255)
            saven = self.log_dir_img + 'dk'+sid+'.png'
            tmp=F.sigmoid(dd01)[0].squeeze().data.cpu().numpy()
            cv2.imwrite(saven,tmp*255)
            self.cnt += 1
        return loss_seg, loss_dd

    def set_log_dir(self, phase, saveid):
        self.epoch = 0
        self.phase = phase
        self.saveid = saveid
        self.log_dir = os.path.join(self.model_dir, "{}_{}".format(phase, saveid))
        self.log_dir_model = self.log_dir +'/'+ 'models'
        if not os.path.exists(self.log_dir_model):
            os.makedirs(self.log_dir_model)
        self.log_dir_img = self.log_dir +'/'+ 'imgs'
        if not os.path.exists(self.log_dir_img):
            os.makedirs(self.log_dir_img)
        self.checkpoint_path_seg = os.path.join(self.log_dir_model, "seg_*epoch*.pth".format())
        self.checkpoint_path_seg = self.checkpoint_path_seg.replace("*epoch*", "{:04d}")
        self.checkpoint_path_ssdd = os.path.join(self.log_dir_model, "ssdd_*epoch*.pth".format())
        self.checkpoint_path_ssdd = self.checkpoint_path_ssdd.replace("*epoch*", "{:04d}")

def models(config, weight_file=None):
    seg_model = SegModel(config=config)
    seg_model.initialize_weights()
    seg_model.load_resnet38_weights(weight_file)
    ssdd_model = SSDDModel(config=config)
    ssdd_model.initialize_weights()
    return (seg_model, ssdd_model)
