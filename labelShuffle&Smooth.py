import os
import torch
import options
import numpy as np
import utility_Func
import AIC_scene_data
from Meter import Meter
from torchvision import transforms
from torch.autograd import Variable
from AIC_scene_data import AIC_scene
from torch.utils.data import DataLoader

if __name__ == "__main__":

    args = options.parse_args()

    loadPath = os.path.join(args.path, args.best_model)
    checkpoint = torch.load(loadPath)
    print("=====> best prec3 : {}".format(checkpoint['best_prec3']))

    model = checkpoint['model']
    model = model.cuda()
    model.eval()

    # ----- supervised label shuffle
    val_data = AIC_scene(
        part="val",
        path=args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale(args.scrop),
            AIC_scene_data.TenCrop(args.scrop),
            AIC_scene_data.ToTensor(eval=True),
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # ImageNet
        ]))
    val_loader = DataLoader(val_data,batch_size=args.batchSize,shuffle=False,num_workers=args.workers)

    top3 = Meter()
    cls_top3 = {i : Meter() for i in range(80)}
    pred = list()

    for ith_batch, data in enumerate(val_loader):

        print(ith_batch,len(val_loader))
        tmp = list()
        final_output = torch.zeros(len(data['label']),80).cuda()
        print(final_output.size())
        for i in range(10):
            input = data['image'][i]
            input = input.cuda()
            input_var = Variable(input)
            output = model(input_var)
            tmp.append(output.data)

        for i in range(len(data['label'])):
            for j in range(10):
                final_output[i,:]+=tmp[j][i,:]
            final_output[i,:].div_(10.0)
        final_outputVar = Variable(final_output)

        _, pred_index = torch.topk(final_outputVar.data, 1, dim=1, largest=True, sorted=True)
        pred.extend(list(pred_index.cpu().numpy()))

        res, cls1, cls3 = utility_Func.accuracy(final_output,data['label'].cuda(),topk=(0, 2))
        top3.update(res[1])
        for i in range(len(data['label'])):
            cls_top3[data['label'][i]].update(cls3[i])

    # get class index of low accuracy
    low = list()
    for i in range(80):
        if cls_top3[i].avg() < top3.avg():
            low.append(i)
    print(len(low))
    utility_Func.supervised_label_shuffle(os.path.join(args.path,"ai_challenger_scene_train_20170904","train_label.txt"),
                                          os.path.join(args.path,"ai_challenger_scene_train_20170904","shuffle_label.txt"),
                                          low)
    np.save("lowAccuracy.npy",low)

    # ----- label smoothing
    utility_Func.confusionMatrix(os.path.join(args.path,"ai_challenger_scene_validation_20170908","val_label.txt"),
                                 pred,val_data.__len__())
    utility_Func.priorLabelDis(args.epsilon,args.t)