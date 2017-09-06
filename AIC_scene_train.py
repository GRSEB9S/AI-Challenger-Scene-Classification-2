from AIC_scene_data import scene_Classification
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.distributed as Distributed

def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(loader):
        imgs_Batch,label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,help="how many Gpus to be used")
    parser.add_argument('--model',default='DenseNet',type=str,help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--batchSize',default=16,type=int,help="batch Size")
    parser.add_argument('--worldSize',default=2,type=int,help="number of distributed processes")
    parser.add_argument('--pretrained',default=True,type=bool,help="whether to use pretrained models or not")
    parser.add_argument('--workers',default=4,type=int,help="number of data loading workers")
    parser.add_argument('--epochs',default=10,type=int,help="number of training epochs")
    args = parser.parse_args()

    # pretrained models
    # DenseNet:densenet_consine_264_k48.py; trained on ImageNet, validated
    # ResNext1101:resnext_101_32_4d.py; trained on ImageNet, validated
    # ResNext2101:resnext_101_64x4d.py; trained on ImageNet, validated
    # ResNext50:resnext_50_32x4d.py; trained on ImageNet, validated
    # ResNet50:resnet50_places365_scratch.py, trained on Places365_standard, unvalidated
    # ResNet152:resnet152_places365_scratch.py, trained on Places365_standard, unvalidated

    models = ['DenseNet','ResNext1101','ResNext2101','ResNext50','ResNet50','ResNet152']
    if args.model not in models: raise ValueError('specify the right model name!')
    models_dict = {'DenseNet' : 'densenet_consine_264_k48',
                   'ResNext1101' : 'resnext_101_32_4d',
                   'ResNext2101' : 'resnext_101_64x4d',
                   'ResNext50' : 'resnext_50_32x4d',
                   'ResNet50' : 'resnet50_places365_scratch',
                   'ResNet152' : 'resnet152_places365_scratch'}
    model_path = "/data/chaoyang/Places_challenge2017/"

    # ---------------------------------------------------
    # multiple Gpu version loading and distributing model
    # ---------------------------------------------------

    if torch.cuda.is_available():

        # load model
        if args.pretrained:
            print("=====> loading pretrained model : {}{}".format(args.pretrained, args.model))
            if args.model == models[0]:
                import densenet_cosine_264_k48
                model = densenet_cosine_264_k48.densenet_cosine_264_k48
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[0]])))
            elif args.model == models[1]:
                import resnext_101_32x4d
                model = resnext_101_32x4d.resnext_101_32x4d
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[1]])))
            elif args.model == models[2]:
                import resnext_101_64x4d
                model = resnext_101_64x4d.resnext_101_64x4d
                model.load_state_dict(torch.load("{}{}.pth", format(model_path, models_dict[models[2]])))
            elif args.model == models[3]:
                import resnext_50_32x4d
                model = resnext_50_32x4d.resnext_50_32x4d
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[3]])))
            elif args.model == models[4]:
                import resnet50_places365_scratch
                model = resnet50_places365_scratch.resnet50_places365
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[4]])))
            else:
                import resnet152_places365_scratch
                model = resnet152_places365_scratch.resnet152_places365
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[5]])))
        else:
        # @TODO how to train non-pretrained model?

        if args.gpus == 1:

            # @TODO  move model to single gpu

        else:

            if args.worldSize == 1:
                raise ValueError('specify at least 2 processes for distributed training')
            Distributed.init_process_group(backend='gloo', init_method=, world_size=args.worldSize)

            model.cuda()
            net = DistributedDataParallel(model,device_ids=list(range(args.gpus))) # output stored in gpus[0]

    else:
        raise ValueError('Gpus not available yet!')

    # ---------------------------------------------------
    #                                        data loading
    # ---------------------------------------------------

    train_dataset = scene_Classification(
        part='train',
        Transform=transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip,
            transforms.ToTensor(),  # pixel values range from 0.0 to 1.0
            # transforms.Normalize(mean=TODO,std=TODO)  calculate mean and std for each image or the whole dataset?
        ]))
    train_Sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_Loader = DataLoader(train_dataset,batch_size=args.batchSize,shuffle=True,num_workers=args.workers,batch_sampler=train_Sampler,drop_last=True)

    # ---------------------------------------------------
    #                                               train
    # ---------------------------------------------------

    # define loss function and optimizer
    criterion = pass
    loss = pass
    optimizer = pass

    for ith_epoch in range(args.epochs):

        for i, (input,label) in enumerate(train_Loader):

            # measure data loading time.

            label_var = torch.autograd.Variable(label.cuda(async=True))
            input_var = torch.autograd.Variable(input)

            # compute output
            output = model(input_var)
            loss = criterion()

            # compute gradient and do a SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

