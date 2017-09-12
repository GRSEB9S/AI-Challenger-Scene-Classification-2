import argparse
import shutil
import time
import AIC_scene_data

import torch.utils.data
import torch.optim as optim
import torch
import self_models
import torch.nn as nn
import torch.cuda
import logging

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import DataParallel
from AIC_scene_data import AIC_scene
from torch.autograd import Variable
from Plot import Plot
from Meter import Meter

def _make_dataloaders(train_set, val_set):

    train_Loader = DataLoader(train_set,
                              batch_size=args.batchSize,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    val_Loader = DataLoader(val_set,
                            batch_size=int(args.batchSize/16),
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            drop_last=True)

    return train_Loader,val_Loader


def _set_lr(optimizer, ith_epoch, epochs):

    # sets the learning rate of initial lr decayed by 10 every 30 epochs
    learning_rate = args.lr * (0.1 ** (ith_epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        print('=====> setting learning_rate to : {},{}/{}'.format(learning_rate, ith_epoch, epochs))

def save_checkpoint(state,path,model_name,is_best):

    checkpoint_path = "{}/{}_{}.pth.tar".format(path,model_name,state['epoch']-1)

    torch.save(state,checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,"{}/{}_best.pth.tar".format(path,model_name))

def train(train_Loader, model, criterion, optimizer, ith_epoch,):

    data_time = Meter() # measure average batch data loading time
    batch_time = Meter() # measure average batch computing time, including forward and backward
    losses = Meter() # record average losses across all mini-batches within an epoch
    prec1 = Meter()
    prec3 = Meter()

    model.train()

    end = time.time()
    for ith_batch, data in enumerate(train_Loader):

        input , label = data['image'], data['label']
        input, label = input.cuda(), label.cuda()
        data_time.update(time.time()-end)
        end = time.time()

        # Forward pass
        input_var = Variable(input)
        label_var = Variable(label)
        output = model(input_var)
        loss = criterion(output, label_var) # average loss within a mini-batch

        # measure accuracy and record loss
        _prec1,_prec3 = accuracy(output.data,label,topk=(0,2))
        losses.update(loss.data[0])
        prec1.update(_prec1)
        prec3.update(_prec3)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 0

        batch_time.update(time.time()-end)
        end = time.time()

        bt_avg,dt_avg,loss_avg,prec1_avg,prec3_avg = batch_time.avg(),data_time.avg(),losses.avg(),prec1.avg(),prec3.avg()
        if ith_batch % args.print_freq == 0:
            print('Train : ith_batch, batches, ith_epoch : %s %s %s\n' %(ith_batch,len(train_Loader),ith_epoch),
                  'Averaged Batch-computing Time : %s \n' % bt_avg,
                  'Averaged Batch-loading Time : %s \n' % dt_avg,
                  'Averaged Batch-Loss : %s \n' % loss_avg,
                  'Averaged Batch-prec1 : %s \n' % prec1_avg,
                  'Averaged Batch-prec3 : %s \n' % prec3_avg)

    return losses.avg()

def validate(val_Loader,model,criterion,ith_epoch):

    data_time = Meter()  # measure average batch data loading time
    batch_time = Meter()  # measure average batch processing time, including forward and output
    losses = Meter()  # record average losses across all mini-batches within an epoch
    top1 = Meter()  # record average top1 precision across all mini-batches within an epoch
    top3 = Meter()  # record average top3 precision

    model.eval()

    end = time.time()
    for ith_batch, data in enumerate(val_Loader):

        input, label = data['image'], data['label']
        input, label = input.cuda(), label.cuda()
        data_time.update(time.time() - end)
        end = time.time()

        # Forward pass
        input_var = Variable(input)
        label_var = Variable(label)
        output = model(input_var)
        loss = criterion(output, label_var)  # average loss within a mini-batch

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, label, topk=(0, 2))
        losses.update(loss.data[0])
        top1.update(prec1)
        top3.update(prec3)

        batch_time.update(time.time() - end)
        end = time.time()

        bt_avg,dt_avg,loss_avg,top1_avg,top3_avg = batch_time.avg(),data_time.avg(),losses.avg(),top1.avg(),top3.avg()
        if ith_batch % args.print_freq == 0:
            print('Validate : ith_batch, batches, ith_epoch : %s %s %s \n' % (ith_batch, len(val_Loader), ith_epoch),
                  'Averaged Batch-computing Time : %s \n' % bt_avg,
                  'Averaged Batch-loading Time : %s \n' % dt_avg,
                  'Averaged Batch-Loss : %s \n' % loss_avg,
                  'Averaged Batch-Prec@1 : %s \n' % top1_avg,
                  'Averaged Batch-Prec@3 : %s \n' % top3_avg)
    return losses.avg(),top1.avg(),top3.avg()

def accuracy(output,label,topk=(0,)):

    # compute accuracy for precision@k for the specified k
    # output : Batch x n_classes

    maxk = max(topk)
    _, pred_index = torch.topk(output,maxk+1,dim=1,largest=True,sorted=True) # descending order
    correct = pred_index.eq(label.view(len(label),-1).expand_as(pred_index))
    for i in range(len(label)):
        for j in range(3):
            if correct[i,j] == 1 :
                for k in range(j+1,3):
                    correct[i,k] = 1
                break
    res=[]
    for k in topk:
        correct_k = float(correct[:,k].sum()) / float(len(label))
        res.append(correct_k)

    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,help="how many Gpus to be used")
    parser.add_argument('--model',default='ResNet152',type=str,help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--batchSize',default=128,type=int,help="batch Size")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--pretrained',default=True,type=bool,help="whether to use pretrained models or not")
    parser.add_argument('--workers',default=8,type=int,help="number of data loading workers")
    parser.add_argument('--epochs',default=80,type=int,help="number of training epochs")
    parser.add_argument('--start-epoch',type=int,default=0,help="start epoch, useful when retraining")
    parser.add_argument('--lr','--learning-rate',type=float,default=0.1,help="learning rate")
    parser.add_argument('--weight-decay',default=1e-4,type=float,help='weight decay')
    parser.add_argument('--print-freq',default=50,type=int,help="print training statics every print_freq batches")
    parser.add_argument('--save-freq',default=10,type=int,help="save checkpoint every save_freq epochs")
    parser.add_argument('--lr-decay',default=15,type=int,help="learning rate decayed every lr_decay epochs")
    parser.add_argument('--resume',default=None,type=str,help="path to model to be resumed")
    parser.add_argument('--path',default="/data/chaoyang/scene_Classification",type=str,help="root path")
    parser.add_argument('--pre_model_path', default="/data/chaoyang/Places_challenge2017/", type=str,
                        help="path to pre-trained models")
    args = parser.parse_args()

    logger = logging.getLogger(args.model)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("{}.log".format(args.model))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(time)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


    # pretrained models
    # DenseNet:densenet_consine_264_k48.py; trained on ImageNet, validated
    # ResNext1101:resnext_101_32_4d.py; trained on ImageNet, validated
    # ResNext2101:resnext_101_64x4d.py; trained on ImageNet, validated
    # ResNext50:resnext_50_32x4d.py; trained on ImageNet, validated
    # ResNet50:resnet50_places365_scratch.py, trained on Places365_standard, unvalidated
    # ResNet152:resnet152_places365_scratch.py, trained on Places365_standard, unvalidated

    pre_models = ['DenseNet', 'ResNext1101', 'ResNext2101', 'ResNext50', 'ResNet50', 'ResNet152','DenseNet161']
    if args.model not in pre_models and args.pretrained == True: raise ValueError('please specify the right pre_trained model name!')
    models_dict = {'DenseNet' : 'densenet_cosine_264_k48',
                   'ResNext1101' : 'resnext_101_32_4d',
                   'ResNext2101' : 'resnext_101_64x4d',
                   'ResNext50' : 'resnext_50_32x4d',
                   'ResNet50' : 'resnet50_places365_scratch',
                   'ResNet152' : 'resnet152_places365_scratch'}
    pre_model_path = args.pre_model_path

    # ---------------------------------------------------
    #                                        data loading
    # ---------------------------------------------------

    train_dataset = AIC_scene(
        part="train",
        path = args.path,
        Transform=transforms.Compose([
            AIC_scene_data.RandomSizedCrop(224),
            AIC_scene_data.RandomHorizontalFlip(),
            AIC_scene_data.ToTensor(),  # pixel values range from 0.0 to 1.0
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet
        ]))
    print(train_dataset.__len__())
    val_dataset = AIC_scene(
        part="val",
        path = args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale(256),
            AIC_scene_data.CenterCrop(224),
            AIC_scene_data.ToTensor(),
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet
        ]))
    print(val_dataset.__len__())
    train_Loader,val_Loader = _make_dataloaders(train_dataset,val_dataset)

    # ---------------------------------------------------
    # multiple Gpu version loading and distributing model
    # ---------------------------------------------------

    if args.resume is None:

        # load model
        if args.pretrained:
            print("=====> loading pretrained model : {}  {}".format(args.pretrained, args.model))
            if args.model == pre_models[0]:
                import densenet_cosine_264_k48
                model = densenet_cosine_264_k48.densenet_cosine_264_k48
            elif args.model == pre_models[1]:
                import resnext_101_32x4d
                model = resnext_101_32x4d.resnext_101_32x4d
            elif args.model == pre_models[2]:
                import resnext_101_64x4d
                model = resnext_101_64x4d.resnext_101_64x4d
            elif args.model == pre_models[3]:
                import resnext_50_32x4d
                model = resnext_50_32x4d.resnext_50_32x4d
            elif args.model == pre_models[4]:
                import resnet50_places365_scratch
                model = resnet50_places365_scratch.resnet50_places365
            else:
                import resnet152_places365_scratch
                model = resnet152_places365_scratch.resnet152_places365

            if args.model == pre_models[6]:
                model = torch.load("{}{}.pth".format(pre_model_path,models_dict[args.model]))
                model.classifier = nn.Linear(2208,80)
            else:
                pre_state_dict = torch.load("{}{}.pth".format(pre_model_path, models_dict[args.model]))
                layers = list(pre_state_dict.keys())
                pre_state_dict.pop(layers[-1])
                pre_state_dict.pop(layers[-2])
                model.load_state_dict(pre_state_dict)

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

            model = DataParallel(model, device_ids=list(range(args.gpus)))  # output stored in gpus[0]
            model = model.cuda()

        global optimizer
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,

                              nesterov=True)
    else:

        print("=====> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec3 = checkpoint['best_prec3']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        train_losses,val_losses = checkpoint['train_losses'],checkpoint['val_losses']
        prec1,prec3 = checkpoint['prec1'],checkpoint['prec3']
        best_prec3 = checkpoint['best_prec3']
        print("=====> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # ---------------------------------------------------
    #                                               train
    # ---------------------------------------------------

    if args.resume is None:
        best_prec3 = 0
        train_losses,val_losses,prec1,prec3 = Meter(),Meter(),Meter(),Meter()

    stats = Plot(args.model)
    for ith_epoch in range(args.start_epoch,args.epochs):

        _set_lr(optimizer, ith_epoch, args.epochs)

        train_loss = train(train_Loader,model,criterion,optimizer,ith_epoch)
        train_losses.update(train_loss)

        # evaluate on validation set
        val_loss,_prec1,_prec3 = validate(val_Loader,model,criterion,ith_epoch)
        print("=====> Validation set : prec@1 : %s \t prec@3 : %s" % (_prec1,_prec3))
        val_losses.update(val_loss)
        prec1.update(_prec1)
        prec3.update(_prec3)

        stats.update_statistics(ith_epoch,train_loss,val_loss,_prec1,_prec3)

        # determine if model is the best
        is_best = _prec3 > best_prec3
        best_prec3 = max(_prec3,best_prec3)

        if ith_epoch % args.save_freq == 0 :
            save_checkpoint({
                'epoch': ith_epoch + 1,
                'model_name': args.model,
                'model': model,
                'best_prec3': best_prec3,
                'optimizer': optimizer,
                'train_losses': train_losses.val, # list
                'val_losses' : val_losses.val,
                'prec1' : prec1.val,
                'prec3' : prec3.val
            }, args.path , args.model, is_best)
        elif is_best:
            print('=====> setting new best precision@3 : {}'.format(best_prec3))
            save_checkpoint({
                'epoch': ith_epoch + 1,
                'model_name': args.model,
                'model': model,
                'best_prec3' : best_prec3,
                'optimizer': optimizer,
                'train_losses': train_losses.val,  # list
                'val_losses': val_losses.val,
                'prec1': prec1.val,
                'prec3': prec3.val
            }, args.path , args.model, is_best)


