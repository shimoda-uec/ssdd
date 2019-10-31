# SSDD:Self-Supervised Difference Detection
By Watal Shimoda and Keiji Yanai.
## Description
This repository contains the codes for "Self-supervised difference detection for weakly supervised segmentation".  
It has been published at ICCV2019.

## Requirements
Pytorch, [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

## Usage
You can test our trained model by following commands.
```
wget http://mm.cs.uec.ac.jp/shimoda-k/models/segmodel_64pt9_val.pth
export voc_root="your voc root path"
python main_ssdd.py --test=1
```
We are also preparing the full training codes and it will be appreared in soon.

## License and Citation
Please cite our paper if it helps your research:
```
@inproceedings{shimodaICCV19  
  Author = {Wataru Shimoda and Keiji Yanai},
  Title = {Self-supervised difference detection for weakly supervised segmentation},  
  Booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  Year = {2019}
}  
```

## Acknowledgment
Many codes of this repository have been derived from [PSA](https://github.com/jiwoon-ahn/psa).
