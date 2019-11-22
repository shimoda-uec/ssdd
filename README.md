# SSDD:Self-Supervised Difference Detection
By Watal Shimoda and Keiji Yanai.  
[paper](https://arxiv.org/abs/1911.01370)

## Description
This repository contains the codes for "Self-Supervised Difference Detection for Weakly Supervised Segmentation".  
It has been published at ICCV2019.  

We define an inputs of the pair of segmentation labels as Knowledge and Advice.  
The proposed method integrates the pair of segmentation labels by Self-Supervised Difference Detection(SSDD) module.  
In the paper, first, we integrate the segmentation labels of Pixel-level Semantic Affinity(PSA) and CRF applied segmentation masks by 
considering the labels as Knowledge and its CRF results as Advice.  
We denote this approach as static SSDD module.  
Furthermore, we develop the labels obtained in the previous step using two SSDD modules,
and we train a segmentation model with the modules in an end-to-end manner.  
We denote this approach as dynamic SSDD module.  
In this dynamic module, we intend to adapt the proposed method to an iterative training approach proposed by Wei et al. : [arxiv](https://arxiv.org/abs/1509.03150).

<img src="https://github.com/shimoda-uec/ssdd/blob/master/figure/ssdd_module.png">

## Visualization
We provide the progress of training and inference on the validation set re-produced with this repository.  
The progress of training for dynamic ssdd module: [html](http://mm.cs.uec.ac.jp/shimoda-k/space0/wseg/ssdd/git/ssdd/script/dssdd.html).  
The inference on the validation: [html](http://mm.cs.uec.ac.jp/shimoda-k/space0/wseg/ssdd/git/ssdd/script/val.html).  
(65.4 in validation. There is only difference in the option of interpolation between this repository and conference version.)  
We also provide trained models, pre-computed results and evaluation results: [run.zip](http://mm.cs.uec.ac.jp/shimoda-k/space0/wseg/ssdd/git/ssdd/run.zip).

## Requirements
Python 3.5 , Pytorch >= 0.4.1, [Pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

## Advance preparation
We assume that the root directory of Pascal VOC is located in the same directory with this name: "voc_root".  
So set a symblic link for the root directory.
```
ln -s "your_voc_root" voc_root
```

To train both of static ssdd module and dynamic ssdd module, seed labels are required.
Please check this directory: [preparing_labels](https://github.com/shimoda-uec/ssdd/tree/master/prepare_labels).  

## Usage
First, train static SSDD module by following codes. (around half day)  
```
python main_ssdd.py --mode=0
```

Second, compute the probability maps of the difference detection in advance. (around 1 hour)  
```
python main_ssdd.py --mode=1
```

Third, train dynamic SSDD module. (around one day)  
```
python main_ssdd.py --mode=2
```

After the training of the dynamic SSDD module, you can test your trained model.  
```
python main_ssdd.py --mode=3
```

## License and Citation
Please cite our paper if it helps your research:
```
@inproceedings{shimodaICCV19,
  Author = {Wataru Shimoda and Keiji Yanai},
  Title = {Self-Supervised Difference Detection for Weakly-Supervised Segmentation},  
  Booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  Year = {2019}
}  
```

## Acknowledgment
Many codes of this repository have been derived from [PSA](https://github.com/jiwoon-ahn/psa).
