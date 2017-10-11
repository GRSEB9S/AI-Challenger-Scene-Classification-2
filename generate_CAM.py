# ResNet50/ResNet152 last_conv_layer 7
# ChampResNet152 last_conv layer 13
# Resolution: scale(256,384,512), scrop(224,336,448) to maintain crop ratio
# last conv feature map according to scale(256,384,512) is (8,12,16), scrops are (7,11,14)
# last conv feature map's channels are all 2048
# need to retrain models that take different scale size as input size
# after generating heatmap, use scropped image region to train other models.

import torch
import sys
import numpy as np
import AIC_scene_data
import cv2
import os
import collections

from torchvision import transforms
from AIC_scene_data import AIC_scene
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F


def hook_feature(module,input,output):

    feature_map.append(output.data)

def return_CAM(conv_feature,weight_softmax,cls_idx):

    size_upsample = {'224':(256,256),'336':(384,384),'448':(512,512)}

    bs, nc, h, w = conv_feature.size()
    cam = torch.mm(weight_softmax[cls_idx].unsqueeze(0),conv_feature.view(nc,h*w))
    cam = cam.cpu().numpy()
    cam = cam.reshape(h, w) # reshape and view is the same order ?
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample[str(scrop)])

    return output_cam

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("please specify the right best model path and last conv name:\n")
        print("example : \t/data/chaoyang/scene_Classification/ResNet50_best_lr0.1_depth32_bs256.pth.tar\n")
        print("\t  last conv layer name scropSize saved_data path crop_nums per image")
        exit()

    checkpoint = torch.load(sys.argv[1])
    print(checkpoint['best_prec3'])
    _state_dict = checkpoint['model'].state_dict()
    state_dict = collections.OrderedDict()
    for i in range(len(_state_dict.keys())):
        old_key = list(_state_dict.keys())[i]
        new_key = old_key.replace("module.", "")
        if old_key in ['module.12.1.weight','module.12.1.bias']:
            new_key = old_key.replace("module.12","13")
        state_dict[new_key] = _state_dict[old_key]
    import resnet50_places365_scratch
    model = resnet50_places365_scratch.resnet50_places365
    model.load_state_dict(state_dict)
    last_conv = sys.argv[2]
    scrop = int(sys.argv[3])
    size = {'224':256,'336':384,'448':512}
    data_path = sys.argv[4]
    crop_nums = int(sys.argv[5])
    if checkpoint['model_name'] in ['ResNet50','ResNet152']:
        assert last_conv == '7'
    elif checkpoint['model_name'] == "ChampResNet152":
        assert last_conv == '13'
    else:
        pass

    # hook last conv layer for feature map
    # note it's a building block module, not the last conv layer in the network
    # for scrop 224
    model._modules.get(last_conv).register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())  # 80 * 2048
    weight_softmax = torch.from_numpy(weight_softmax).cuda()

    model.eval()
    model = model.cuda(device_id=0)

    train_dataset = AIC_scene(
        part='train',
        path='/data/chaoyang/scene_Classification',
        Transform=transforms.Compose([
            AIC_scene_data.Scale((scrop,scrop)),
            AIC_scene_data.CenterCrop(scrop),
            AIC_scene_data.ToTensor(),
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    train_Loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=8)

    wrong = 0 # record number of wrong prediction images
    # crop_probs = list()
    index = list()
    coordinates = list()
    for i,data in enumerate(train_Loader):
        print(i/len(train_Loader))
        input,label = data['image'].cuda(),data['label'].numpy() # torch.cuda.LongTensor size 1
        input_Var = Variable(input)

        feature_map = [] # 1 * nc * h * w
        output = model(input_Var) # batchSize(1) * n_classes

        norm_output = F.softmax(output).data.squeeze() # normalize each row so that probability sums up to 1
        probs, idx = torch.sort(norm_output,dim=0,descending=True)
        idx = idx.cpu().numpy()
        # generate Class Activation Mapping recursively before predicted right until top3
        correct = False
        for j in range(3):
            if idx[j] == label :
                prob_conf = list()
                CAM = return_CAM(feature_map[0],weight_softmax,idx[j])
                for l in range(np.power(size[str(scrop)] - scrop + 1, 2)):
                    r = l // (size[str(scrop)]-scrop+1)
                    c = l % (size[str(scrop)]-scrop+1)
                    prob_conf.append(np.sum(CAM[r:r+scrop,c:c+scrop]))
                coordinate = np.argsort(prob_conf)[-crop_nums:] # probs is an ascending order, returning array
                # crop_probs.append(CAM)
                index.append(i)
                coordinates.append(coordinate)
                correct = True
                break

        if correct == False:
            wrong += 1

        print(wrong)

    print(wrong)
    np.savez(os.path.join(data_path,"crop_probs.npz"),index=index,coordinates=coordinates)




