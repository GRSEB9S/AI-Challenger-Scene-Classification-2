import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class LSR(nn.Module):
    def __init__(self):
        self.priorDis = np.load("priorDis.npy")

    def forward(self,input,target):
        # input target are Variable(batchSize,80),(batchSize)
        label = Variable(torch.Tensor(len(target),80)) #(batchSize,80)
        batch_loss = Variable(torch.Tensor(len(target)))
        for i in range(len(target)):
            label[i] = torch.from_numpy(np.array(self.priorDis[target.data[i]]))
            batch_loss[i] = torch.mul(torch.sum(torch.mul(torch.log(input[i]),label[i])),-1)

        return torch.mean(batch_loss)
