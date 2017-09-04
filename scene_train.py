from AIC_scene_data import scene_Classification
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import matplotlib.pyplot as plt
import resnet152_places365_scratch
import torch
import torch.nn as nn

def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(loader):
        imgs_Batch,label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser
    parser.add_argument("--which",default="train",help="specify which part of dataset")
    parser.parse_args()

    dataset = scene_Classification(
        part=parser.which,
        Transform=transforms.Compose([
        ]))

    loader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4,drop_last=True)

    # it's the converted pytorch resnet152 trained on Places365_Standard from scratch
    resnet152_places365_scratch = resnet152_places365_scratch.resnet152_places365
    resnet152_places365_scratch.load_state_dict(torch.load("/data/chaoyang/Places_challenge2017/resnet152_places365.pth"))

    if torch.cuda.is_available():

        torch.cuda.set_device(0)
        resnet152_places365_scratch = resnet152_places365_scratch.cuda()



