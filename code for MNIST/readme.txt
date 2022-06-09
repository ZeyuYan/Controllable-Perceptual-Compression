## Optimally Controllable Perceptual Lossy Compression

This folder contains the training and test codes for the MNIST experiment of this paper.
This code for MNIST is a faithful reimplementation of (https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN) in PyTorch.

* download the dataset 
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/


## Development Environment

* Win10
* NVIDIA GeForce RTX 2080
* cuda version 11.1
* Python 3.6
* pytorch 1.2.0
* torchvision 0.4.0
* matplotlib 3.1.2

# Training code: 
The two stages of training as a whole are included in the code "train.py"

# Testing code
test.py

# Parameters can be modified in train.py or test.py in the section：
-- training parameters
batch_size = 128
lr = 0.001
train_epoch = 100
lambda_gp = 10
beta = 0.99
pretrained = False
rate = 4

--testing parameters
batch_size = 100
pretrained = True
rate = 4

## Citation
@inproceedings{2022ControllablePerceptualCompression,
	author={Zeyu Yan, Fei Wen, Peilin Liu},
	booktitle={Proceedings of the
	International Conference on Machine Learning (ICML)},
	title={Optimally Controllable Perceptual Lossy Compression},
	year={2022}, 
}