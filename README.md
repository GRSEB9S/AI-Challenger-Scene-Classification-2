# Scene Classification for AI Challenger

This repository is hosted for [**Scene Classification of AI Challenger competition**](https://challenger.ai/competition/scene) since it comes to an end. The final accuracy we achieved on testA dataset is appromaximately 96.5%, with a large margin behind the champion, which is around 98.8%. We have been not paying attention and energy into this competition for over a month due to other business, since it's over, we make it publicly available. We implemented multiple tricks using pytorch, which is crutial to competition, if you're interested in competitions related to scene classification, this may help. Detailed description of usages are listed as follows.

## Training a convolutional neural network

The main python script is `AIC\_scene\_train.py`, provides functions to train and validate your neural network plus plotting the coressponding curves using tensorboardX. The default training is using gpu, do make sure that you've install cuda\cudnn\nvidia driver, etc. You can also carry on the distributed training, however, given official ImageNet distributed training example from pytorch, it doesn't seem to work. 

To training the CNN, simply type following command in your shell:

> python3 AIC\_scene\_train.py --batchSize 256 --epochs 30 --stepSize 0.2 --lr 0.1

For more options, please check `options.py`	

## Label smoothing regularization 

It was used in [**Rethinking the Inception Architecture for Computer Vision**](https://arxiv.org/abs/1512.00567) to reduce model's overfitting. There are some hyperparameters need to be set:

