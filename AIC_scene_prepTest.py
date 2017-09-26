import argparse
import torch
import AIC_scene_data
import json
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="prepare json file for test")
    parser.add_argument("--model",default="/data/chaoyang/scene_Classification/ResNet50_best_lr0.1_depth32_bs256.pth.tar",
                        type=str,help="abs path for testing model")
    parser.add_argument("--path",default="/data/chaoyang/scene_Classification",type=str,help="root path")
    parser.add_argument("--scrop",default=(224,224))
    args = parser.parse_args()

    checkpoint = torch.load(args.model)
    print("==========> val_best_prec3 %s" % checkpoint['best_prec3'])
    model = checkpoint['model']

    AIC_scene_testA = AIC_scene_data.AIC_scene_test(
        part="testA",
        path=args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale(args.scrop),
            AIC_scene_data.CenterCrop(args.scrop[0]),
            AIC_scene_data.ToTensor(),
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet
        ]))

    testA_Loader = DataLoader(AIC_scene_testA,batch_size=4,shuffle=False,num_workers=8)

    record = list()
    with open(os.path.join(args.path,"ai_challenger_scene_test_a_20170922/pred.json"),'w') as f:
        for i,data in enumerate(testA_Loader):
            print(i,len(testA_Loader))

            input,img_name = data['image'].cuda(),data['label']
            input_Var = Variable(input)
            print(img_name[0])
            print(img_name[-1])

            output = model(input_Var) # batchSize x n_classes

            _, pred_index = torch.topk(output,k=3,dim=1,largest=True,sorted=True)  # descending order
            pred_index = pred_index.data.cpu().numpy()
            for j in range(input.size(0)):
                record.append({"label_id":list(pred_index[j,:]),"image_id":img_name[j]})
        json.dump(record,f)




