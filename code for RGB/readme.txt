## Optimally Controllable Perceptual Lossy Compression

This folder contains the training and testing codes of the proposed Framework A for the RGB experiment of this paper.
This code for RGB is a faithful reimplementation of (https://hific.github.io) in PyTorch.

The model is trained on the COCO2014 dataset
Note that we have not uploaded the trained models since they largly exceeds the maximum permitted size.

## Development Environment
* Win10
* NVIDIA GeForce RTX 2080
* cuda version 11.1
* Python 3.6
* pytorch 1.2.0
* torchvision 0.4.0
* matplotlib 3.1.2
More detials are in requirements.txt

##training code######################################################
The training procedure contains two stages, as describled in Section 4.

train.py     for stage-1 training
train2.py   for stage-2 training

Stage-1): First, an MMSE model is trained by:
python train.py --model_type compression --regime low --n_steps 1e6 
This model is saved in ./Experiment/

Stage-2):  Then, freeze the Encoder and train the decoder $G_p$ with freezed Encoder by:
python train2.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart -ckpt ./model/MMSE_model.pt

The decoder $G_p$ is saved in ./checkpoint/

##testing code######################################################
For testing, set the address of $G_p$ to ./compress2.py, line 262, and run:
python compress2.py -i \path\to\test\dataset -ckpt ./model/MMSE_model.pt --reconstruct

## Citation
@inproceedings{2022ControllablePerceptualCompression,
	author={Zeyu Yan, Fei Wen, Peilin Liu},
	booktitle={Proceedings of the
	International Conference on Machine Learning (ICML)},
	title={Optimally Controllable Perceptual Lossy Compression},
	year={2022}, 
}