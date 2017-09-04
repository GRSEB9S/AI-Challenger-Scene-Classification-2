from AIC_scene_data import scene_Classification
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import matplotlib.pyplot as plt
import resnet152_places365_scratch
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(loader):
        imgs_Batch,label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--which',default="train",type=str,help="specify which part of dataset")
    args = parser.parse_args()

    dataset = scene_Classification(
        part=args.which,
        Transform=transforms.Compose([
            transforms.Scale((224,224)),
            transforms.ToTensor(), # pixel values range from 0.0 to 1.0
            transforms.Normalize(mean=#TODO,std=#TODO) # calculate mean and std for each image or the whole dataset?
        ]))

    loader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4,drop_last=True)

    # it's the converted pytorch resnet152 trained on Places365_Standard from scratch
    resnet152_places365_scratch = resnet152_places365_scratch.resnet152_places365
    resnet152_places365_scratch.load_state_dict(torch.load("/data/chaoyang/Places_challenge2017/resnet152_places365.pth"))

    if torch.cuda.is_available():

        torch.cuda.set_device(0)
        resnet152_places365_scratch = resnet152_places365_scratch.cuda()

    criterion = nn.Softmax()
    optimizer = optim.SGD(resnet152_places365_scratch.parameters(),lr=0.001,momentum=0.9)

    # for epoch in range(100):
        # running_loss = 0.0

