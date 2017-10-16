import options
import shutil
import time
import AIC_scene_data
import Plot
import math
import os

import os
import torch.utils.data
import torch.optim as optim
import torch
import self_models
import torch.nn as nn
import torch.cuda

from Meter import Meter
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.nn import DataParallel
from AIC_scene_data import AIC_scene
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from labelSmooth_Loss import LSR

def _make_dataloaders(train_set, val_set):

    train_Loader = DataLoader(train_set,
                              batch_size=args.batchSize,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True)

    val_Loader = DataLoader(val_set,
                            batch_size=int(args.batchSize/8),
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True)

    return train_Loader,val_Loader

def _set_lr(optimizer, ith_epoch, epochs, cosine=False):

    # sets the learning rate of initial lr decayed by 10 every 30 epochs
    if cosine:
        learning_rate = 0.5 * args.lr * (1 + math.cos(math.pi * ith_epoch / epochs))
    else:
        learning_rate = args.lr * (args.stepSize ** (ith_epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('=====> setting learning_rate to : {},{}/{}'.format(learning_rate, ith_epoch, epochs))

def _save_checkpoint(state,path,model_name,lr,depth,batchsize,scale,lrdecay,gpus,optimizer,is_best):

    checkpoint_path = "{}/{}_{}_lr{}_depth{}_bs{}_scale{}_lrdecay{}_gpus{}_optimizer{}.pth.tar".\
        format(path,model_name,state['epoch']-1,lr,depth,batchsize,scale,lrdecay,gpus,optimizer)

    torch.save(state,checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,"{}/{}_best_lr{}_depth{}_bs{}_scale{}_lrdecay{}_gpus{}_optimizer{}.pth.tar".
                        format(path,model_name,lr,depth,batchsize,scale,lrdecay,gpus,optimizer))

def train(train_Loader,model,criterion,optimizer,ith_epoch):

    data_time = Meter() # measure average batch data loading time
    batch_time = Meter() # measure average batch computing time, including forward and backward
    losses = Meter() # record average losses across all mini-batches within an epoch
    prec1 = Meter()
    prec3 = Meter()

    super_label = list()
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
        res, cls1, cls3 = accuracy(output.data,label,False,topk=(0,2))
        losses.update(loss.data[0])
        prec1.update(res[0])
        prec3.update(res[1])

        for i in range(args.batchSize):
            if cls3[i] != 1:
                super_label.append(data['idx'][i])

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

    return losses.avg(),prec1.avg(),prec3.avg()

def validate(val_Loader,model,criterion,ith_epoch):

    batch_time = Meter()  # measure average batch processing time, including forward and output
    losses = Meter()  # record average losses across all mini-batches within an epoch
    top1 = Meter()  # record average top1 precision across all mini-batches within an epoch
    top3 = Meter()  # record average top3 precision
    cls_top1,cls_top3 = dict(),dict()
    for i in range(80):
        cls_top1[i] = Meter()
        cls_top3[i] = Meter()

    model.eval()
    end = time.time()
    for ith_batch, data in enumerate(val_Loader):

        # Forward pass
        tmp = list()
        final_output = torch.zeros(args.batchSize // 8, 80).cuda()
        for i in range(10):
            input = data['image'][i]
            input = input.cuda()
            input_var = Variable(input)
            output = model(input_var)  # args.batchSize //8  x 80
            tmp.append(output.data)

        for i in range(args.batchSize//8):
            for j in range(10):
                final_output[i,:]+=tmp[j][i,:]
            final_output[i,:].div_(10.0)
        final_output_var = Variable(final_output)
        loss = criterion(final_output_var, Variable(data['label'].cuda()))  # average loss within a mini-batch

        # measure accuracy and record loss
        res, cls1, cls3 = accuracy(final_output,data['label'].cuda(),True,topk=(0, 2))
        prec1,prec3 = res[0],res[1]
        losses.update(loss.data[0])
        top1.update(prec1)
        top3.update(prec3)
        for i in range(args.batchSize//8):
            cls_top1[data['label'][i]].update(cls1[i])
            cls_top3[data['label'][i]].update(cls3[i])

        batch_time.update(time.time() - end)
        end = time.time()

        bt_avg,loss_avg,top1_avg,top3_avg = batch_time.avg(),losses.avg(),top1.avg(),top3.avg()
        if ith_batch % args.print_freq == 0:
            print('Validate : ith_batch, batches, ith_epoch : %s %s %s \n' % (ith_batch, len(val_Loader), ith_epoch),
                  'Averaged Batch-computing Time : %s \n' % bt_avg,
                  'Averaged Batch-Loss : %s \n' % loss_avg,
                  'Averaged Batch-Prec@1 : %s \n' % top1_avg,
                  'Averaged Batch-Prec@3 : %s \n' % top3_avg)

    return losses.avg(),top1.avg(),top3.avg(),cls_top1,cls_top3

def accuracy(output,label,val=False,topk=(0,)):

    # compute accuracy for precision@k for the specified k and each class
    # output : BatchSize x n_classes

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

    cls1, cls3 = list(), list()
    for i in range(len(label)):
       cls1.append(correct[i,0])
       cls3.append(correct[i,2])
    return res, cls1, cls3

    return res, cls1, cls3

if __name__ == '__main__':

    args = options.parse_args()

    # pretrained models
    # DenseNet:densenet_consine_264_k48.py; trained on ImageNet, validated
    # ResNext1101:resnext_101_32_4d.py; trained on ImageNet, validated
    # ResNext2101:resnext_101_64x4d.py; trained on ImageNet, validated
    # ResNext50:resnext_50_32x4d.py; trained on ImageNet, validated
    # ResNet50:resnet50_places365_scratch.py, trained on Places365_standard, unvalidated
    # ResNet152:resnet152_places365_scratch.py, trained on Places365_standard, unvalidated

    pre_models = ['DenseNet', 'ResNext1101', 'ResNext2101', 'ResNext50', 'ResNet50', 'ResNet152','DenseNet161','ChampResNet152','ResNet50AIC80','ResNet50GWAP','ResNet50MeanMax']
    if args.model not in pre_models and args.pretrained == True: raise ValueError('please specify the right pre_trained model name!')
    models_dict = {'DenseNet' : 'densenet_cosine_264_k48',
                   'ResNext1101' : 'resnext_101_32x4d',
                   'ResNext2101' : 'resnext_101_64x4d',
                   'ResNext50' : 'resnext_50_32x4d',
                   'ResNet50' : 'resnet50_places365_scratch',
                   'ResNet152' : 'resnet152_places365_scratch',
                   'ChampResNet152' : 'Places2_365_CNN'}
    pre_model_path = args.pre_model_path
    crop_dict = {224:256,320:395}
    # ---------------------------------------------------
    #                                        data loading
    # ---------------------------------------------------


    train_dataset = AIC_scene(
        part="train",
        path = args.path,
        Transform=transforms.Compose([
            # AIC_scene_data.RandomScaleCrop(),
            AIC_scene_data.RandomSizedCrop(args.scrop),
            # AIC_scene_data.supervised_Crop((args.scrop,args.scrop),os.path.join(args.path,"AIC_train_scrop224")),
            AIC_scene_data.RandomHorizontalFlip(),
            AIC_scene_data.ToTensor(),  # pixel values range from 0.0 to 1.0
            # AIC_scene_data.Normalize(mean=[0.4951, 0.476, 0.4457],
            #                          std=[0.2832, 0.2788, 0.2907]) # 3 x 224 x 224
            # AIC_scene_data.Normalize(mean=[0.4952, 0.476 0.4457],
            #                          std=[0.2858, 0.2814, 0.2931]) # 3 x 336 x 336
            # AIC_scene_data.Normalize(mean=[0.4927, 0.4735, 0.4435],
            #                          std=[0.2884, 0.2839, 0.2952])
        ]))
    print(train_dataset.__len__())
    val_dataset = AIC_scene(
        part="val",
        path = args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale(crop_dict[args.scrop]),
            AIC_scene_data.TenCrop(args.scrop),
            AIC_scene_data.ToTensor(eval=True),
            AIC_scene_data.Normalize(mean=[0.4956, 0.4791, 0.4486],
                                     std=[0.2822, 0.2779, 0.2905],  # 3 x 224 x 224
                                     eval=True)  # ImageNet # return list per image
            # AIC_scene_data.Normalize(mean=[0.4956, 0.4791, 0.4487],
            #                          std=[0.2848, 0.2805, 0.2929],
            #                          eval=True) # 3 x 336 x 336
            # AIC_scene_data.Normalize(mean=[0.4956, 0.4791, 0.4487],
            #                          std=[0.2861, 0.2818, 0.2941],
            #                          eval=True) # 3 x 448 x 448
        ]))
    print(val_dataset.__len__())
    train_Loader,val_Loader = _make_dataloaders(train_dataset,val_dataset)

    writer = SummaryWriter(
        log_dir="runs/{}_lr{}_bs{}_depth{}_lrdecay{}_stepSize{}_gpus{}_scale{}_optimizer{}".format(
            args.model, args.lr, args.batchSize,args.depth,args.lr_decay,args.stepSize,args.gpus,args.scrop,args.optimizer))

    # display only a batch of image without undermining program's speed
    for i,data in enumerate(train_Loader):
        batch_img,batch_label = data['image'],data['label']
        grid = utils.make_grid(batch_img,nrow=16,padding=0,normalize=True)
        writer.add_image('1stbatch_trainImgs',grid)
        if i == 0:
            break

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
            elif args.model == pre_models[5]:
                import resnet152_places365_scratch
                model = resnet152_places365_scratch.resnet152_places365
            elif args.model == pre_models[7]:
                import Places2_365_CNN
                model = Places2_365_CNN.resnet152_places365
            elif args.model == pre_models[8]:
                import resnet50_places365_aic80
                model = resnet50_places365_aic80.resnet50_places365
            elif args.model == pre_models[9]:
                import resnet50_places365_gwap
                model = resnet50_places365_gwap.resnet50_places365
            elif args.model == pre_models[10]:
                import resnet50_places365_meanmax
                model = resnet50_places365_meanmax.resnet50_places365

            if args.model == 'pyResNet50':
                model = torch.load("{}whole_resnet50_places365.pth.tar".format(pre_model_path))
                model.classifier = nn.Linear(2208,80)
            elif args.model == pre_models[7]:
                state_dict = torch.load("{}{}.pth".format(pre_model_path, models_dict[args.model]))
                model.load_state_dict(state_dict)
            elif args.model == pre_models[8]:
                pre_state_dict = torch.load("{}{}.pth".format(pre_model_path, models_dict['ResNet50']))
                model_dict = model.state_dict()
                model_dict.update(pre_state_dict)
                model.load_state_dict(model_dict)
            elif args.model in [pre_models[9], pre_models[10]]:
                pre_state_dict = torch.load("{}{}.pth".format(pre_model_path, models_dict['ResNet50']))
                layers = list(pre_state_dict.keys())
                pre_state_dict.pop(layers[-1])
                pre_state_dict.pop(layers[-2])
                model_dict = model.state_dict()
                model_dict.update(pre_state_dict)
                model.load_state_dict(model_dict) 
            else:
                pre_state_dict = torch.load("{}{}.pth".format(pre_model_path, models_dict[args.model]))
                layers = list(pre_state_dict.keys())
                pre_state_dict.pop(layers[-1])
                pre_state_dict.pop(layers[-2])
                model_dict = model.state_dict()
                model_dict.update(pre_state_dict)
                model.load_state_dict(model_dict)

        else:

            print("=====> create model : {}".format(args.model))

            if args.model == 'DenseNetEfficient':
                model = self_models.DenseNetEfficient()
            elif args.model == 'DenseNetEfficientMulti':
                model = self_models.DenseNetEfficientMulti()
            elif args.model == 'ResNet50':
                import resnet50_places365_scratch
                model = resnet50_places365_scratch.resnet50_places365
            else:
                model = self_models.DenseNetVOC()

        if args.gpus == 1:

            model.cuda()

        else:

            model = DataParallel(model, device_ids=list(range(args.gpus)))  # output stored in gpus[0]
            model = model.cuda()

        # fix certain layers according to args.fine_tune
        # for resnet50: optional depth is : 2,32,87,124,150,153(fine-tune all)
        # for resnet152: optional depth is : 2,32,359,434,464,467(fine-tune all)
        # for resnext50: optional depth is : 2,32,
        # for resnext1101 : optional depth is :
        # for resnext2101 : optional depth is :
        # for ChampResNet152 : optional depth is : 2, 32(conv5),242(conv4),353(conv3),428(conv2),437(fine-tune all)
        if args.depth != 1:
            param_name = list([name for name,_ in model.named_parameters()])
            model_params = list()
            for name, param in model.named_parameters():
                if name == param_name[len(param_name)-args.depth]:
                    break
                else:
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if name in param_name[-args.depth:]:
                    model_params.append({'params':param})

        global optimizer
        if args.optimizer == "SGD":
            optimizer = optim.SGD(model_params if args.depth !=1 else model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=True)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(model_params if args.depth !=1 else model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
    else:

        print("=====> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch'] + 1
        best_prec3 = checkpoint['best_prec3']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        print("=====> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    # define loss function and optimizer
    criterion = nn.LSR().cuda()

    # ---------------------------------------------------
    #                                               train
    # ---------------------------------------------------

    if args.resume is None:
        best_prec3 = 0


    for ith_epoch in range(args.start_epoch,args.epochs):

        # _set_lr(optimizer, ith_epoch, args.epochs)
        AIC_scene_data.label_shuffle(os.path.join(args.path,"ai_challenger_scene_train_20170904","train_label.txt"),
                                     os.path.join(args.path,"ai_challenger_scene_train_20170904","shuffle_label.txt"),
                                     train_dataset,
                                     part="train",path=args.path,
                                     sub_path="ai_challenger_scene_train_20170904",
                                     img_path="scene_train_images_20170904")

        if args.optimizer == 'Adam':
            pass
        else:
            _set_lr(optimizer, ith_epoch, args.epochs, args.cosine)

        train_loss, _train_prec1, _train_prec3 = train(train_Loader,model,criterion,optimizer,ith_epoch)
        writer.add_scalar('train_loss', train_loss, ith_epoch)
        writer.add_scalar('train_prec1', _train_prec1, ith_epoch)
        writer.add_scalar('train_prec3', _train_prec3, ith_epoch)

        # evaluate on validation set
        val_loss, _val_prec1, _val_prec3, val_cls1, val_cls3 = validate(val_Loader, model, criterion, ith_epoch)
        print("=====> Validation set : prec@1 : %s \t prec@3 : %s" % (_val_prec1, _val_prec3))
        writer.add_scalar('val_loss', val_loss, ith_epoch)
        writer.add_scalar('val_prec1', _val_prec1, ith_epoch)
        writer.add_scalar('val_prec3', _val_prec3, ith_epoch)
        for i in train_dataset.id2eng.keys():
            writer.add_scalar("{}_cls1".format(train_dataset.id2eng[i]), val_cls1[int(i)].avg(), ith_epoch)
            writer.add_scalar("{}_cls3".format(train_dataset.id2eng[i]), val_cls3[int(i)].avg(), ith_epoch)

        # determine if model is the best
        is_best = _val_prec3 > best_prec3
        best_prec3 = max(_val_prec3, best_prec3)

        if ith_epoch % args.save_freq == 0 :
            _save_checkpoint({
                'epoch': ith_epoch + 1,
                'model_name': args.model,
                'model': model,
                'best_prec3': best_prec3,
                'optimizer': optimizer,
            },  args.path , args.model, args.lr,args.depth,args.batchSize,
                args.scrop,args.lr_decay,args.gpus,args.optimizer,is_best)
        elif is_best:
            print('=====> setting new best precision@3 : {}'.format(best_prec3))
            _save_checkpoint({
                'epoch': ith_epoch + 1,
                'model_name': args.model,
                'model': model,
                'best_prec3' : best_prec3,
                'optimizer': optimizer,
            },  args.path ,args.model,args.lr,args.depth,args.batchSize,
                args.scrop,args.lr_decay,args.gpus,args.optimizer,is_best)

        for name,param in model.named_parameters():
            writer.add_histogram(name,param.clone().cpu().data.numpy(),ith_epoch)

    writer.close()
