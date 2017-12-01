import os
import torch
import random
import itertools
import numpy as np
import matplotlib.pylab as plt
plt.switch_backend("qt5agg")
from sklearn.metrics import confusion_matrix

def accuracy(output,label,topk=(0,)):

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

def confusionMatrix(shuffle, pred, length):

    y = np.zeros((length))
    with open(shuffle) as f:
        lines = f.readlines()
        for i in range(length):
            y[i] = lines[i].split(" ")[1]
    y_pred = np.array(pred)
    assert len(y_pred) == length
    classes = np.array(list([i for i in range(80)]))
    c = confusion_matrix(y,y_pred,classes)
    c = c.astype('float')/c.sum(axis=1)[:,np.newaxis]
    print(c)
    np.save("NorconfusionMatrix.npy",c)

def plotConfusionMatrix():

    c = np.load("NorconfusionMatrix.npy")
    plt.imshow(c, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(80)
    plt.xticks(tick_marks, np.array(list([i for i in range(80)])), rotation=45)
    plt.yticks(tick_marks, np.array(list([i for i in range(80)])))
    thres = c.max() / 2.
    for i, j in itertools.product(range(c.shape[0]), range(c.shape[1])):
        plt.text(j, i, format(c[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if c[i, j] > thres else 'black')
    plt.tight_layout()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    plt.savefig("confusionMatrix.jpeg")

def priorLabelDis(epsilon,t):

    c = np.load("NorconfusionMatrix.npy")
    assert c.shape == (80,80)
    print(np.sort(c))
    print(np.argsort(c))

    priorDis = list()
    for i in range(80):
        similar = np.argsort(c[i,:])[-3:-1]
        u = list([epsilon/80 for k in range(80)])
        for j in similar:
            u[j] = t/4
        u[i] = 1-t-((80-2)/80)*epsilon
        priorDis.append(u)
    np.save("priorDis.npy",priorDis)

def supervised_label_shuffle(train,shuffle,low,self=None,args=None):

    with open(train,'r') as f:
        lines = f.readlines()
        cls_id = np.zeros(len(lines), dtype=np.int8)
        cls_num, cls_idx, final = list(), list(), list()
        for i in range(len(lines)):
            cls_id[i] = lines[i].split(" ")[1]
        for i in range(80):
            cls_num.append(len(np.where(cls_id == i)[0]))
            cls_idx.append(list(np.argwhere(cls_id == i)[:, 0]))
        maximum, minimum = max(cls_num), min(cls_num)
        idx = list(range(maximum))
        for i in range(80):
            if i in low:
                random.shuffle(idx)
                random_idx = [k.item() for k in np.mod(idx, cls_num[i])]
                this = cls_idx[i]
                for j in range(maximum):
                    final.append(this[random_idx[j]])
            else:
                final.extend(cls_idx[i])
        random.shuffle(final)

    shuff_lines= list()
    with open(shuffle,'w') as f:
        for i in range(len(final)):
            shuff_lines.append(lines[final[i]])
        f.writelines(shuff_lines)

        """
        add_lines = list()
        for i in low:
            random.shuffle(idx)
            random_idx = [k.item() for k in np.mod(idx, cls_num[i])]
            this = cls_idx[i]
            for j in range(maximum):
                add_lines.append(lines[this[random_idx[j]]])
        for i in low:
            this = cls_idx[i]
            for j in this:
                del lines[j]

        random.shuffle(add_lines)
        lines.extend(add_lines)

        with open(shuffle, 'w') as f:
            f.writelines(lines)
        """
    if self != None:
        with open(self.read) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                img_name, label_index = lines[i].split(' ')
                self.image.append(os.path.join(args.path,"ai_challenger_scene_train_20170904","scene_train_images_20170904", img_name))
                self.label.append(int(label_index))

def label_shuffle(train,shuffle,self=None,args=None):

    with open(train,'r') as f:
        lines = f.readlines()
        cls_id = np.zeros(len(lines), dtype=np.int8)
        cls_num, cls_idx, final = list(), list(), list()
        for i in range(len(lines)):
            cls_id[i] = lines[i].split(" ")[1]
        for i in range(80):
            cls_num.append(len(np.where(cls_id == i)[0]))
            cls_idx.append(list(np.argwhere(cls_id == i)[:, 0]))
        maximum, minimum = max(cls_num), min(cls_num)
        idx = list(range(maximum))
        for i in range(80):
            random.shuffle(idx)
            random_idx = [k.item() for k in np.mod(idx, cls_num[i])]
            this = cls_idx[i]
            for j in range(maximum):
                final.append(this[random_idx[j]])
        random.shuffle(final)

    shuff_lines= list()
    with open(shuffle,'w') as f:
        for i in range(len(final)):
            shuff_lines.append(lines[final[i]])
        f.writelines(shuff_lines)

    if self != None:
        with open(self.read) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                img_name, label_index = lines[i].split(' ')
                self.image.append(os.path.join(args.path,"ai_challenger_scene_train_20170904","scene_train_images_20170904", img_name))
                self.label.append(int(label_index))

