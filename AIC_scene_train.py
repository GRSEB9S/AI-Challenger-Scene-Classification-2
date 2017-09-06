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

class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0


    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)


    def value(self):
        if self.cum:
            return self._total
        else:
            return self._total / self._count


    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])

# @Todo 
def _make_dataloaders(train_set, valid_set, test_set, train_size, valid_size, batch_size):
    # Split training into train and validation
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices)-valid_size][:train_size or None]
    valid_indices = indices[len(indices)-valid_size:] if valid_size else None

    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=batch_size)
    if valid_size:
        valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader

# @Todo
def _set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    elif float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])


def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

        # Forward pass
        input_var = Variable(input, volatile=(not train)).cuda(async=True)
        target_var = Variable(target, volatile=(not train), requires_grad=False).cuda(async=True)
        output_var = model(input_var)
        loss = criterion(output_var, target_var)

        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        # Accounting
        _, predictions_var = torch.topk(output_var, 1)
        error = 1 - torch.eq(predictions_var, target_var).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        print('  '.join([
            '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="scene_classification for AI Challenge")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,help="how many Gpus to be used")
    parser.add_argument('--model',default='DenseNet',type=str,help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--batchSize',default=16,type=int,help="batch Size")
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--worldSize',default=2,type=int,help="number of distributed processes")
    parser.add_argument('--pretrained',default=True,type=bool,help="whether to use pretrained models or not")
    parser.add_argument('--workers',default=4,type=int,help="number of data loading workers")
    parser.add_argument('--epochs',default=10,type=int,help="number of training epochs")
    parser.add_argument('--save',default='checkpoint',type=str,help="path to save the model")
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

    torch.manual_seed(0) # for stable result

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
            print("=====> create model : {}".format(args.model))
            if args.model == 'DenseNetEfficient':
                from models import DenseNetEfficient # Single GPU
                model = DenseNetEfficient(# @Todo)
            elif args.model == 'DenseNetEfficientMulti':
                from models import DenseNetEfficientMulti # Multi GPU
                model = DenseNetEfficientMulti(# @Todo)


        if args.gpus == 1:

            # @TODO  move model to single gpu
            model.cuda()

        else:

            if args.worldSize == 1:
                raise ValueError('specify at least 2 processes for distributed training')
            Distributed.init_process_group(backend='gloo', init_method=, world_size=args.worldSize)

            model.cuda() # ??
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
    criterion = nn.CrossEntropyLoss()
    loss = pass
    optimizer = # @Todo optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=True)

    best_error = 1    
    for ith_epoch in range(args.epochs):

        _set_lr(optimizer, epoch, n_epochs, lr)
        train_results = run_epoch(
            loader=train_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=True,
        )
        valid_results = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
        )

        # Determine if model is the best
        _, _, valid_error = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'model.t7'))
