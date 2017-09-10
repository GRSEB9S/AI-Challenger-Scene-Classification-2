import argparse
import torch
import AIC_scene_data

from torch.nn import DataParallel
from AIC_scene_data import scene_testA
from torchvision import transforms

def accuracy(output,label,topk=(0,)):

    # compute accuracy for precision@k for the specified k
    # output : Batch x n_classes

    assert output.shape[0]==args.batchSize

    maxk = max(topk)
    _, pred_index = torch.topk(output,maxk+1,dim=1,largest=True,sorted=True) # descending order
    correct = pred_index.eq(label.view(args.batchSize,-1).expand_as(pred_index))
    for i in range(args.batchSize):
        for j in range(3):
            if correct[i,j] == 1 :
                for k in range(j+1,3):
                    correct[i,k] = 1
                break
    res=[]
    for k in topk:
        correct_k = correct[:,k].sum() / args.batchSize
        res.append(correct_k)

    return res

if __name__ == "__main__" :

    parser =argparse.ArgumentParser
    parser.add_argument('--model',default="ResNet152",type=str,help="which model to test")
    parser.add_argument('--gpuids',default=[0],type=tuple,help="gpu id used to test")
    args = parser.parse_args()

    path = "/data/chaoyang/scene_Classification/ai_challenger_scene_train_20170904/"

    model = torch.load("{}{}_best.pth.tar".format(path,args.model))

    if torch.cuda.is_available():

        if len(args.gpuids) == 1:
            model = model.cuda()
        else:
            model = DataParallel(model,device_ids=args.gpuids).cuda()
    else:
        raise ValueError("gpus not available yet")

    testA_dataset = scene_testA(
            Transform=transforms.Compose([
                AIC_scene_data.Scale(256),
                AIC_scene_data.CenterCrop(224),
                AIC_scene_data.ToTensor(),
                AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) # ImageNet
        ]))

