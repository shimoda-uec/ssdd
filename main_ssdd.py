import os
import time
import numpy as np
import torch.nn as nn
import torch
import ssdd_val as val
#import ssdd_test as test
import train_dssdd 
import train_sssdd 
ROOT_DIR = os.getcwd()
VOC_ROOT = os.environ['voc_root']
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class Config():
    OUT_SHAPE = (112,112)
    INP_SHAPE = (448,448)
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 2e-4
    NUM_CLASSES = 21 
    LEARNING_RATE=1e-3

############################################################
#  Dataset
############################################################

class PascalDataset():
    def load(self):
        image_dir = VOC_ROOT +'/JPEGImages'
        fn='data/trainaug_id.txt'
        f = open(fn,'r')
        image_ids = f.read().splitlines()
        f.close()
        self.image_ids=image_ids
        label_listn='data/trainaug_labels.txt'
        label_list=np.loadtxt(label_listn)
        label_dic={}
        for i in range(len(image_ids)):
          label=label_list[i]
          label_dic[image_ids[i]]=label_list[i]
        self.label_dic=label_dic
    def load_val(self):
        image_dir = VOC_ROOT +'/JPEGImages'
        fn= VOC_ROOT +'/ImageSets/Segmentation/val.txt'
        f = open(fn,'r'); image_ids = f.read().splitlines(); f.close()
        self.image_ids=image_ids
    def load_test(self):
        image_dir = VOC_ROOT +'/JPEGImages'
        fn=VOC_ROOT +'/ImageSets/Segmentation/test.txt'
        f = open(fn,'r');image_ids = f.read().splitlines(); f.close()
        self.image_ids=image_ids


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--test', required=False,
                        default=0,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=int)
    parser.add_argument('--bn', required=False,
                        default=1,
                        metavar="<gpun>",
                        type=int)
    parser.add_argument('--saveid', required=True,
                        default='tmp',
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=str)
    args = parser.parse_args()

    def create_model(config, modellib, modeln, weight_file=None):
        model_factory = modellib.__dict__[modeln]
        model_params = dict(config=config, weight_file=weight_file)
        model = model_factory(**model_params)
        return model

    config = Config()
    config.VOC_ROOT=VOC_ROOT
    runner_name = os.path.basename(__file__).split(".")[0]
    if args.test==0:
        dataset_train=PascalDataset()
        dataset_train.load()
        weight_file=''
        models=create_model(config, train_dssdd, 'models', weight_file)
        config.BATCH=torch.cuda.device_count()*args.bn
        config.EPOCHS=41
        config.saveid=args.saveid
        model_trainer=train_dssdd.Trainer(config=config, model_dir=DEFAULT_LOGS_DIR, model=models)
        model_trainer.set_log_dir(runner_name, 'dssdd', args.saveid)
        model_trainer.train_model(
                    dataset_train,
                    )
    elif args.test==1:
            dataset_val=PascalDataset()
            dataset_val.load_val()
            weight_file='./models/segmodel_64pt9_val.pth'
            print(weight_file)
            model=create_model(config, val, 'val')
            model=nn.DataParallel(model).cuda()
            state_dict = torch.load(weight_file)
            model.load_state_dict(state_dict,strict=False)
            model_evaluator=val.Evaluator(config=config, model=model)
            model_evaluator.config.BATCH=torch.cuda.device_count()*args.bn
            model_evaluator.config.saveid=args.saveid
            model_evaluator.set_log_dir(runner_name, 'val', args.saveid)
            model_evaluator.eval_model(
                        dataset_val,
                        )
