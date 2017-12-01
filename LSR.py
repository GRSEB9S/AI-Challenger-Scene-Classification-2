import torch
import numpy as np
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable

class LSR(Module):
    def __init__(self):
        super(LSR,self).__init__()
        self.priorDis = np.load("priorDis.npy")

    def forward(self,input,target):

        # input is Variable(batchSize,80)
        # target is torch.cuda.Tensor() of shape (batchSize)
        label = Variable(torch.Tensor(len(target),80).cuda()) #(batchSize,80)
        for i in range(len(target)):
            label[i] = torch.from_numpy(np.array(self.priorDis[target[i]]))

        batch_loss = torch.mul(torch.sum(torch.mul(F.log_softmax(input),label),1),-1)

        return torch.mean(batch_loss)
