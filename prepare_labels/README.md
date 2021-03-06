# Preparing labels
## Description
To reproduce the proposed method, preparing seed labels are required.  
For the seed labels, the proposed method uses an existing weakly-supervised segmentation method: Pixel-level Semantic Affinity(PSA): [arxiv](https://arxiv.org/abs/1803.10464).  
The codes in this directory are for generation of the seed labels.  
That are based on the [original implementation](https://github.com/jiwoon-ahn/psa).  

## Usage
The prepararion of the seed labels are consist of tow steps.
First, obtain semantic probality maps using a classification model with CAM.  
Second, generate the seed labels by propagating semantic information using an affinity model.  
Each step requires a trained model, then download the trained models from following links.  
Classification model: [link](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view)  
Affinity model: [link](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view)  
Note that these links are derived from the author's [repository](https://github.com/jiwoon-ahn/psa)  
We recomend to save the models in this direcory: [pretrained_models](https://github.com/shimoda-uec/ssdd/tree/master/pretrained_models).  

After downloding the models, we can prepare the seed labels by following commands.  
```
python make_cam_labels.py  
```
```
python make_aff_labels.py
```
