# Preparing labels
## Description
To reproduce the proposed method, we need to prepare seed labels for traning.
For the seed labels, we use an existing weakly-supervised segmentation method: Pixel-level Semantic Affinity(PSA)[link](https://arxiv.org/abs/1803.10464).
The codes in this directory are for generation of the seed labels.
You can also use the [original implementation](https://github.com/jiwoon-ahn/psa).

## Usage
We prepare the seed labels with tow steps.
First, we obtain semantic probality maps using a classification model with CAM.
Second, we generate the seed labels by propagating semantic information using an affinity model.
Each step requires a trained model, then please download the trained models from following links.
[Classification model](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view)
[Affinity model](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view)
Note that these links are derived from the author's [repository](https://github.com/jiwoon-ahn/psa)
We recomend to save the models in ../pretrained_models.

After downloding the models, we can prepare the seed labels by following commands.
```
python make_cam_labels.py
python make_aff_labels.py
```
