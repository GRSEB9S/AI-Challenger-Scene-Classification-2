import argparse
import torch

def parse_args():

    parser = argparse.ArgumentParser(description="scene classification for AI challenge")
    parser.add_argument('--t',type=float,default=0.01)
    parser.add_argument('--hue',type=float,default=0.4  )
    parser.add_argument('--cosine',type=str,default=False)
    parser.add_argument('--epsilon',type=float,default=0.008)
    parser.add_argument('--contrast',default=0.4,type=float)
    parser.add_argument('--saturation',default=0.4,type=float)
    parser.add_argument('--distributed',type=str,default=False)
    parser.add_argument('--brightness',default=0.4,type=float)
    parser.add_argument('--momentum',default=0.9,type=float,help="momentum")
    parser.add_argument('--batchSize',default=256,type=int,help="batch Size")
    parser.add_argument('--optimizer',default="SGD",type=str,help="optimizer type")
    parser.add_argument('--weight-decay',default=1e-4,type=float,help="weight decay")
    parser.add_argument('--epochs',default=30,type=int,help="number of training epochs")
    parser.add_argument('--stepSize',default=0.2,type=float,help="lr decayed by stepSize")
    parser.add_argument('--resume',default=None,type=str,help="path to model to be resumed")
    parser.add_argument('--scrop',default=224,type=int,help="resolution of training images")
    parser.add_argument('--workers',default=8,type=int,help="number of data loading workers")
    parser.add_argument('--lr','--learning-rate',type=float,default=0.1, help="learning rate")
    parser.add_argument('--depth',default=1,type=int,help="fine tune depth,starting from last layer")
    parser.add_argument('--start-epoch',type=int,default=0,help="start epoch, useful when retraining")
    parser.add_argument('--save-freq',default=10,type=int,help="save checkpoint every save_freq epochs")
    parser.add_argument('--path',default="/data/chaoyang/scene_Classification",type=str,help="root path")
    parser.add_argument('--model',default='ResNet152',type=str,help="which model:DenseNet,ResNext,ResNet")
    parser.add_argument('--lr-decay',default=5,type=int,help="learning rate decayed every lr_decay epochs")
    parser.add_argument('--gpus',default=torch.cuda.device_count(),type=int,help="how many Gpus to be used")
    parser.add_argument('--pretrained',default=True,type=bool,help="whether to use pretrained models or not")
    parser.add_argument('--print-freq', default=50, type=int, help="print training statics every print_freq batches")
    parser.add_argument('--pre_model_path',default="/data/chaoyang/Places_challenge2017/",type=str,help="path to pre-trained models")
    parser.add_argument('--best-model',type=str,default="ResNet50_best_lr0.01_depth1_bs160_scale224_lrdecay5_gpus2_optimizerSGD.pth.tar",help="best model name")
    args = parser.parse_args()

    return args

if __name__ == "__main":

    parse_args()