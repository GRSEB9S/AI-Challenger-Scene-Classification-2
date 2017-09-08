import argparse
import shutil
import matplotlib.pyplot as plt
import time

import torch.utils.data
import torch.optim as optim
import torch.distributed as Distributed
import torch
import self_models
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.nn.parallel import DistributedDataParallel
from AIC_scene_data import scene_Classification
from torch.autograd import Variable

def image_Show():

    plt.clf()
    figure = plt.figure()

    for ith_Batch, sample_Batched in enumerate(train_Loader):
        imgs_Batch , label_Batch = sample_Batched['image'], sample_Batched['label']
        grid = utils.make_grid(imgs_Batch) # make a grid of mini-batch images
        plt.imshow(grid)

class Meter:
    """Computes and stores the average and total value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _make_dataloaders(train_set, val_set):

    if args.gpus > 1:
        train_Sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_Sampler=None

    train_Loader = DataLoader(train_set,
                              batch_size=args.batchSize,
                              shuffle=(train_Sampler is None),
                              num_workers=args.workers,
                              batch_sampler=train_Sampler,
                              pin_memory=True)

    val_Loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True)

    return train_Loader,val_Loader


def _set_lr(optimizer, ith_epoch, epochs):

    # sets the learning rate of initial lr decayed by 10 every 30 epochs
    learning_rate = args.lr * (0.1 ** (ith_epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        print('=====> setting learning_rate to : {},{}/{}'.format(learning_rate, ith_epoch, epochs))

def save_checkpoint(state,model_name,is_best):

    root_path = "{}{}".format(path,"ai_Challenger_scene_train_20170904")
    checkpoint_path = "{}/{}_{}_{}_{}.pth.tar".format(root_path,model_name,state['epoch'],state['lr'],state['weight_decay'])

    torch.save(state,checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,"{}/{}_best.pth.tar".format(root_path,model_name))

def train(train_Loader, model, criterion, optimizer, ith_epoch,):

    data_time = Meter() # measure average batch data loading time
    batch_time = Meter() # measure average batch computing time, including forward and backward
    losses = Meter() # record average losses across all mini-batches within an epoch
    top1 = Meter() # record average top1 precision across all mini-batches within an epoch
    top3 = Meter() # record average top3 precision

    model.train()

    end = time.time()
    for ith_batch, (input, label) in enumerate(train_Loader):

        data_time.update(time.time()-end)
        end = time.time()

        # Forward pass
        label = label.cuda(async=True)
        input_var = Variable(input)
        label_var = Variable(label)
        output = model(input_var)
        loss = criterion(output, label_var) # average loss within a mini-batch

        # measure accuracy and record loss
        prec1,prec3 = accuracy(output.data,label,topk=(1,3))
        losses.update(loss.data[0])
        top1.update(prec1.data[0])
        top3.update([prec3.data[0]])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 0

        batch_time.update(time.time()-end)
        end = time.time()

        if ith_batch % args.print_freq == 0:
            print('Train : ith_batch, batches, ith_epoch : {}/{}/{}\t'
                  'Averaged Batch-computing Time : {batch_time.avg:.5f}\t'
                  'Averaged Batch-loading Time : {data_time.avg:.5f}\t'
                  'Averaged Batch-Loss : {losses.avg:.4f}\t'
                  'Averaged Batch-Prec@1 : {top1.avg:.4f}\t'
                  'Averaged Batch-Prec@3 : {top3.avg:.4f}'.format(
                  ith_batch,len(train_Loader),ith_epoch,batch_time,data_time,losses,top1=top1,top3=top3
            ))

    return losses.avg

def validate(val_Loader,model,criterion,ith_epoch):

    data_time = Meter()  # measure average batch data loading time
    batch_time = Meter()  # measure average batch processing time, including forward and output
    losses = Meter()  # record average losses across all mini-batches within an epoch
    top1 = Meter()  # record average top1 precision across all mini-batches within an epoch
    top3 = Meter()  # record average top3 precision

    model.val()

    end = time.time()
    for ith_batch, (input, label) in enumerate(train_Loader):

        data_time.update(time.time() - end)
        end = time.time()

        # Forward pass
        label = label.cuda(async=True)
        input_var = Variable(input)
        label_var = Variable(label)
        output = model(input_var)
        loss = criterion(output, label_var)  # average loss within a mini-batch

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, label, topk=(1, 3))
        losses.update(loss.data[0])
        top1.update(prec1.data[0])
        top3.update([prec3.data[0]])

        batch_time.update(time.time() - end)
        end = time.time()

        if ith_batch % args.print_freq == 0:
            print('Test : ith_batch, batches, ith_epoch{}/{}/{}\t'
                  'Averaged Batch-processing Time : {batch_time.avg:.5f}\t'
                  'Averaged Batch-loading Time : {data_time.avg:.5f}\t'
                  'Averaged Batch-Loss : {losses.avg:.4f}\t'
                  'Averaged Batch-Prec@1 : {top1.avg:.4f}\t'
                  'Averaged Batch-Prec@3 : {top3.avg:.4f}'.format(
                ith_batch,len(val_Loader),ith_epoch,batch_time,data_time,losses,top1,top3
            ))

    return top1.avg,top3.avg

def accuracy(output,label,topk=(1,)):

    # compute accuracy for precision@k for the specified k

    maxk = max(topk)
    _, pred_index = torch.topk(output,maxk,dim=1,largest=True,sorted=True) # descending order
    correct = pred_index.eq(label.view(args.batchSize,-1).expand_as(pred_index))

    res=[]
    for k in topk:
        correct_k = correct[:,:k].view(-1).float().mean()
        res.append(correct_k)

    return res

if __name__ == '__main__':

    global path
    path = "/data/chaoyang/scene_Classification/"  # path to your saved model,same as datasets dir

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,help="how many Gpus to be used")
    parser.add_argument('--model',default='DenseNet',type=str,help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--batchSize',default=128,type=int,help="batch Size")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--worldSize',default=4,type=int,help="number of distributed processes")
    parser.add_argument('--pretrained',default=True,type=bool,help="whether to use pretrained models or not")
    parser.add_argument('--workers',default=4,type=int,help="number of data loading workers")
    parser.add_argument('--epochs',default=60,type=int,help="number of training epochs")
    parser.add_argument('--start-epoch',type=int,default=0,help="start epoch, useful when retraining")
    parser.add_argument('--save',default='checkpoint',type=str,help="path to save the model")
    parser.add_argument('--lr','--learning-rate',type=float,default=0.1,help="learning rate")
    parser.add_argument('--weight-decay',default=1e-4,type=float,help='weight decay')
    parser.add_argument('--print-freq',default=10,type=int,help="print training statics every print_freq batches")
    parser.add_argument('--lr-decay',default=20,type=int,help="learning rate decayed every lr_decay epochs")
    args = parser.parse_args()

    # pretrained models
    # DenseNet:densenet_consine_264_k48.py; trained on ImageNet, validated
    # ResNext1101:resnext_101_32_4d.py; trained on ImageNet, validated
    # ResNext2101:resnext_101_64x4d.py; trained on ImageNet, validated
    # ResNext50:resnext_50_32x4d.py; trained on ImageNet, validated
    # ResNet50:resnet50_places365_scratch.py, trained on Places365_standard, unvalidated
    # ResNet152:resnet152_places365_scratch.py, trained on Places365_standard, unvalidated

    pre_models = ['DenseNet', 'ResNext1101', 'ResNext2101', 'ResNext50', 'ResNet50', 'ResNet152']
    if args.model not in pre_models and args.pretrained == True: raise ValueError('please specify the right pre_trained model name!')
    models_dict = {'DenseNet' : 'densenet_cosine_264_k48',
                   'ResNext1101' : 'resnext_101_32_4d',
                   'ResNext2101' : 'resnext_101_64x4d',
                   'ResNext50' : 'resnext_50_32x4d',
                   'ResNet50' : 'resnet50_places365_scratch',
                   'ResNet152' : 'resnet152_places365_scratch'}
    pre_model_path = "/data/chaoyang/Places_challenge2017/"

    torch.manual_seed(0) # for stable result

    # ---------------------------------------------------
    # multiple Gpu version loading and distributing model
    # ---------------------------------------------------

    if torch.cuda.is_available():

        # load model
        if args.pretrained:
            print("=====> loading pretrained model : {}  {}".format(args.pretrained, args.model))
            if args.model == pre_models[0]:
                import densenet_cosine_264_k48
                model = densenet_cosine_264_k48.densenet_cosine_264_k48
                model.load_state_dict(torch.load("{}{}.pth".format(pre_model_path, models_dict[pre_models[0]])))
            elif args.model == pre_models[1]:
                import resnext_101_32x4d
                model = resnext_101_32x4d.resnext_101_32x4d
                model.load_state_dict(torch.load("{}{}.pth".format(pre_model_path, models_dict[pre_models[1]])))
            elif args.model == pre_models[2]:
                import resnext_101_64x4d
                model = resnext_101_64x4d.resnext_101_64x4d
                model.load_state_dict(torch.load("{}{}.pth", format(pre_model_path, models_dict[pre_models[2]])))
            elif args.model == pre_models[3]:
                import resnext_50_32x4d
                model = resnext_50_32x4d.resnext_50_32x4d
                model.load_state_dict(torch.load("{}{}.pth".format(pre_model_path, models_dict[pre_models[3]])))
            elif args.model == pre_models[4]:
                import resnet50_places365_scratch
                model = resnet50_places365_scratch.resnet50_places365
                model.load_state_dict(torch.load("{}{}.pth".format(pre_model_path, models_dict[pre_models[4]])))
            else:
                import resnet152_places365_scratch
                model = resnet152_places365_scratch.resnet152_places365
                model.load_state_dict(torch.load("{}{}.pth".format(pre_model_path, models_dict[pre_models[5]])))
        else:

            print("=====> create model : {}".format(args.model))

            if args.model == 'DenseNetEfficient':
                model = self_models.DenseNetEfficient()
            elif args.model == 'DenseNetEfficientMulti':
                model = self_models.DenseNetEfficientMulti()
            else:
                raise ValueError('please specify the right created self_model name')

        if args.gpus == 1:

            model.cuda()

        else:

            if args.worldSize == 1:
                raise ValueError('specify at least 2 processes for distributed training')
            Distributed.init_process_group(backend='gloo', init_method="tcp://10.65.1.181:22",
                                           world_size=args.worldSize,rank=0)

            net = DistributedDataParallel(model.cuda(),device_ids=list(range(args.gpus))) # output stored in gpus[0]

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

    val_dataset = scene_Classification(
        part='val',
        Transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize()
        ]))

    train_Loader,val_Loader = _make_dataloaders(train_dataset,val_dataset)


    # ---------------------------------------------------
    #                                               train
    # ---------------------------------------------------

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=True)

    best_prec3 = 0
    losses = Meter() # record loss of training epochs

    for ith_epoch in range(args.start_epoch,args.epochs):

        _set_lr(optimizer, ith_epoch, args.epochs)

        loss_avg = train(train_Loader,model,criterion,optimizer,ith_epoch)
        losses.update(loss_avg,len(train_Loader))

        # evaluate on validation set
        prec1,prec3 = validate(val_Loader,model,criterion,ith_epoch)

        # determine if model is the best
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3,best_prec3)
        if is_best:
            print('=====> setting new best precision@3 : {}'.format(best_prec3))

        save_checkpoint({
            'epoch': ith_epoch + 1,
            'model_name': args.model,
            'state_dict': model.state_dict(),
            'best_prec3' : best_prec3,
            'optimizer': optimizer.state_dict()
        }, is_best)
