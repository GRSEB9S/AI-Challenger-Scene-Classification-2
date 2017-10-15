import argparse
import torch
import AIC_scene_data
import json
import os

from torch.nn import DataParallel
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="prepare json file for test")
    parser.add_argument("--model",default="/data/chaoyang/scene_Classification/ResNet50_best_lr0.1_depth32_bs256.pth.tar",
                        type=str,help="abs path for testing model")
    parser.add_argument("--path",default="/data/chaoyang/scene_Classification",type=str,help="root path")
    parser.add_argument("--subpath",default="ai_challenger_scene_test_a_20170922")
    parser.add_argument("--scrop",default=224,type=int)
    parser.add_argument("--save",type=str)
    parser.add_argument("--batchsize",default=4,type=int)

    args = parser.parse_args()

    crop_dict = {224:256,320:395}

    checkpoint = torch.load(args.model)
    print("==========> val_best_prec3 %s" % checkpoint['best_prec3'])
    model = checkpoint['model']
    model.eval()

    AIC_scene_testA = AIC_scene_data.AIC_scene_test(
        part="testA",
        path=args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale(crop_dict[args.scrop]),
            AIC_scene_data.TenCrop(args.scrop),
            AIC_scene_data.ToTensor(eval=True),
            AIC_scene_data.Normalize(mean=[0.4951, 0.476, 0.4457],
                                     std=[0.2832, 0.2788, 0.2907],
				                     eval=True)
        ]))

    testA_Loader = DataLoader(AIC_scene_testA,batch_size=args.batchsize,shuffle=False,num_workers=8)

    record = list()
    with open(os.path.join("submit","{}_testA.json".format(args.save)),'w') as f:
        for ith_batch, data in enumerate(testA_Loader):
            print(ith_batch,len(testA_Loader))

            tmp = list()
            final_output = torch.zeros(args.batchsize, 80).cuda()
            for i in range(10):
                input = data['image'][i]
                input = input.cuda()
                input_var = Variable(input)
                output = model(input_var)
                tmp.append(output.data)

            input,img_name = data['image'].cuda(),data['label']
            input_Var = Variable(input)

            for i in range(args.batchsize):
                for j in range(10):
                    final_output[i,:]+=tmp[j][i,:]
                final_output[i,:].div_(10.0)

            img_name = data['label']
            print(img_name[0])
            print(img_name[-1])

            _, pred_index = torch.topk(final_output,k=3,dim=1,largest=True,sorted=True)  # descending order
            pred_index = pred_index.cpu().numpy()
            for j in range(input.size(0)):
                record.append({"label_id":[k.item() for k in pred_index[j,:]],"image_id":img_name[j]})
        json.dump(record,f)




