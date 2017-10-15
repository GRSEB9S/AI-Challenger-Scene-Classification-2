import torch
import itertools
import numpy as np
import AIC_scene_data
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from AIC_scene_data import AIC_scene
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

def _confusionMatrix(shuffle_label):
    pred = list()
    for ith_batch,data in enumerate(train_Loader):
        print(ith_batch,len(train_Loader))
        input,label = data['image'].cuda(),data['label'].cuda()
        input_var = Variable(input)
        output = model(input_var)
        _,pred_index = torch.topk(output.data,1,dim=1,largest=True,sorted=True)
        pred.extend(list(pred_index.cpu().numpy()))

    y = np.zeros((train_dataset.__len__()))
    with open(shuffle_label) as f:
        lines = f.readlines()
        for i in range(train_dataset.__len__()):
            y[i] = lines[i].split(" ")[1]
    y_pred = np.array(pred)
    assert len(y_pred) == train_dataset.__len__()
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

def priorLabelDis():

    c = np.load("NorconfusionMatrix.npy")
    assert c.shape == (80,80)
    print(np.sort(c))
    print(np.argsort(c))

    priorDis = list()
    for i in range(80):
        sort = np.sort(c[i,:])[-5:-1]
        similar = np.argsort(c[i,:])[-5:-1]
        u = list([epsilon/80 for i in range(80)])
        for j in similar:
            u[j] = t/4
        u[i] = 1-t-((80-4)/80)*epsilon
        priorDis.append(u)
    print(u)
    np.save("priorDis.npy",priorDis)

if __name__ == "__main__":

    epsilon = 0.008
    t = 0.01
    model_path = "/data/chaoyang/scene_Classification/ResNet50_best_lr0.01_depth1_bs160_scale224_lrdecay5_gpus2_optimizerSGD.pth.tar"
    shuffle_label = "/data/chaoyang/scene_Classification/ai_challenger_scene_train_20170904/shuffle_label.txt"
    checkpoint = torch.load(model_path)
    print("=====> best_prec3 : {}".format(checkpoint['best_prec3']))
    model = checkpoint['model']
    model = DataParallel(model, device_ids=list(range(2)))
    model = model.cuda()

    train_dataset = AIC_scene(
        part="train",
        path="/data/chaoyang/scene_Classification",
        Transform=transforms.Compose([
            # AIC_scene_data.RandomScaleCrop(),
            AIC_scene_data.RandomSizedCrop(224),
            # AIC_scene_data.supervised_Crop((args.scrop,args.scrop),os.path.join(args.path,"AIC_train_scrop224")),
            AIC_scene_data.RandomHorizontalFlip(),
            AIC_scene_data.ToTensor(),  # pixel values range from 0.0 to 1.0
            AIC_scene_data.Normalize(mean=[0.4951, 0.476, 0.4457],
                                     std=[0.2832, 0.2788, 0.2907])
        ]))

    train_Loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    _confusionMatrix(shuffle_label)
    priorLabelDis()
