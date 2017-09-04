from AIC_scene_data import scene_Classification
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import matplotlib.pyplot as plt

def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(loader):
        imgs_Batch,label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)
        plt.pause(3)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser
    parser.add_argument("--which",default="train",help="specify which part of dataset")
    parser.parse_args()

    dataset = scene_Classification(
        part=parser.which,
        Transform=transforms.Compose([
        ]))

    loader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4,drop_last=True)


