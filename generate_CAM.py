# ResNet50/ResNet152 last_conv_layer 7
# ChampResNet152 last_conv layer 13
# Resolution: scale(256,384,512), scrop(224,336,448) to maintain crop ratio
# last conv feature map according to scale(256,384,512) is (8,12,16), scrops are (7,11,14)
# last conv feature map's channels are all 2048
# need to retrain models that take different scale size as input size
# after generating heatmap, use scropped image region to train other models.

import os
import cv2
import torch
import options
import numpy as np
import collections
import AIC_scene_data
from torch.nn import DataParallel
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from AIC_scene_data import AIC_scene
from torch.utils.data import DataLoader


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
    output_cam = cv2.resize(cam_img, size_upsample[str(args.scrop)])

    return output_cam

if __name__ == "__main__":

    args = options.parse_args()

    checkpoint = torch.load(os.path.join(args.path,args.best_model))
    print("=====> best_prec3 is : {}".format(checkpoint['best_prec3']))

    _state_dict = checkpoint['model'].state_dict()
    state_dict = collections.OrderedDict()
    for ith_batch in range(len(_state_dict.keys())):
        old_key = list(_state_dict.keys())[ith_batch]
        new_key = old_key.replace("module.", "")
        if old_key in ['module.12.1.weight','module.12.1.bias']:
            new_key = old_key.replace("module.12","13")
        state_dict[new_key] = _state_dict[old_key]
    import resnet50_places365_scratch
    model = resnet50_places365_scratch.resnet50_places365
    model.load_state_dict(state_dict)
    size = {'224':256,'336':384,'448':512}
    if checkpoint['model_name'] in ['ResNet50','ResNet152']:
        assert args.last_conv == '7'
    elif checkpoint['model_name'] == "ChampResNet152":
        assert args.last_conv == '13'
    else:
        pass

    # hook last conv layer for feature map
    # note it's a building block module, not the last conv layer in the network
    # for scrop 224
    model._modules.get(args.last_conv).register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())  # 80 * 2048
    weight_softmax = torch.from_numpy(weight_softmax).cuda()

    model = DataParallel(model,device_ids=list(range(args.gpus)))
    model = model.cuda()

    train_data = AIC_scene(
        part='train',
        path=args.path,
        Transform=transforms.Compose([
            AIC_scene_data.Scale((args.scrop,args.scrop)),
            AIC_scene_data.CenterCrop(args.scrop),
            AIC_scene_data.ToTensor(),
            AIC_scene_data.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet
        ]))
    train_loader = DataLoader(train_data,batch_size=args.batchSize,shuffle=False,num_workers=8)

    wrong = 0 # record number of wrong prediction images
    index = list()
    coordinates = list()

    for ith_batch, data in enumerate(train_loader):
        print(ith_batch / len(train_loader))
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
                for l in range(np.power(size[str(args.scrop)] - args.scrop + 1, 2)):
                    r = l // (size[str(args.scrop)]-args.scrop+1)
                    c = l % (size[str(args.scrop)]-args.scrop+1)
                    prob_conf.append(np.sum(CAM[r:r+args.scrop,c:c+args.scrop]))
                coordinate = np.argsort(prob_conf)[-args.crop_nums:] # probs is an ascending order, returning array
                # crop_probs.append(CAM)
                index.append(ith_batch)
                coordinates.append(coordinate)
                correct = True
                break

        if correct == False:
            wrong += 1

        print(wrong)

    print(wrong)
    np.savez(os.path.join(args.path,"crop_probs.npz"),index=index,coordinates=coordinates)




