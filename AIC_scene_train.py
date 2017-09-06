import argparse
import shutil
import matplotlib.pyplot as plt

import torch.utils.data
import torch.optim as optim
import torch.distributed as Distributed
import torch

from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.nn.parallel import DistributedDataParallel
from AIC_scene_data import scene_Classification


def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(loader):
        imgs_Batch,label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)

def save_checkpoint(state,model_name,is_best):

    root_path = "{}/{}".format(path,"ai_Challenger_scene_train_20170904")
    checkpoint_path = "{}/{}_{}_{}_{}.pth.tar".format(root_path,model_name,state['epoch'],state['lr'],state['weight_decay'])

    torch.save(state,checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,"{}/{}_best.pth.tar".format(root_path,model_name))

def train(train_Loader,model,criterion,optimizer,ith_epoch):

    for ith_batch, (input, label) in enumerate(train_Loader):
        # measure data loading time.

        label_var = torch.autograd.Variable(label.cuda(async=True))
        input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        loss = criterion(output,label_var)

        # compute gradient and do a SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training information
        if ith_batch % args.print_freq == 0:

            print('Epoch : ')


def adjust_lr(optimizer,ith_epoch):

    lr = pass
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,
                        help="how many Gpus to be used")
    parser.add_argument('--model',default='DenseNet',type=str,
                        help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--batchSize',default=16,type=int,
                        help="mini-batch Size")
    parser.add_argument('--worldSize',default=4,type=int,
                        help="number of distributed processes participated in distributed training")
    parser.add_argument('--pretrained',default=True,type=bool,
                        help="whether to use pretrained models or not")
    parser.add_argument('--workers',default=4,type=int,
                        help="number of data loading workers")
    parser.add_argument('--start-epoch',default=0,type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--epochs',default=10,type=int,
                        help="number of training epochs")
    parser.add_argument('--lr','learning -rate',default=0.1,type=float,
                        help="initial learning rate")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum : 0.9")
    parser.add_argument('--weight-decay',default=1e-4,type=float,help="weight decay for decreasing weights")
    parser.add_argument('--print-freq',default=10,type=int,
                        help="print training information per print_freq mini-Batches")
    args = parser.parse_args()

    # pretrained models
    # DenseNet:densenet_consine_264_k48.py; trained on ImageNet, validated
    # ResNext1101:resnext_101_32_4d.py; trained on ImageNet, validated
    # ResNext2101:resnext_101_64x4d.py; trained on ImageNet, validated
    # ResNext50:resnext_50_32x4d.py; trained on ImageNet, validated
    # ResNet50:resnet50_places365_scratch.py, trained on Places365_standard, unvalidated
    # ResNet152:resnet152_places365_scratch.py, trained on Places365_standard, unvalidated

    path = "/data/chaoyang/scene_Classification"
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
                model.load_state_dict(torch.load("{}{}.pth".format(model_path, models_dict[models[2]])))
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

            #init_method is one of your machines ip address plus port.
            Distributed.init_process_group(backend='gloo',
                                           init_method='tcp://10.65.1.181:22',
                                           world_size=args.worldSize)

            model.cuda()
            model = DistributedDataParallel(model,device_ids=list(range(args.gpus))) # output stored in gpus[0]

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
    train_Loader = DataLoader(train_dataset,
                              batch_size=args.batchSize,
                              shuffle=True,
                              num_workers=args.workers,
                              batch_sampler=train_Sampler,
                              drop_last=True)

    # ---------------------------------------------------
    #                                               train
    # ---------------------------------------------------

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),lr=args.lr,
                          momentum=args.momentum,weight_decay=args.weight_decay)

    for ith_epoch in range(args.start_epoch,args.epochs):

        if args.gpus > 1:
            train_Sampler.set_epoch(ith_epoch)

        train(train_Loader,model,criterion,optimizer,ith_epoch)

        save_checkpoint({
            'epoch' : ith_epoch + 1,
            'model_name' : args.model,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict()
        }, is_best)

