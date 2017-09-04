# specify the "path" variable pointing to your overall dir of dataset

from torch.utils.data import Dataset
from skimage import io
import csv

class scene_Classification(Dataset):

    global path, which,id2chi,id2eng
    path = "/data/chaoyang/scene_Classification"
    which = ["train","val","testA","testB"]
    id2chi, id2eng = dict(), dict()

    def __init__(self,part="train",Transform=None):

        with open("{}/{}".format(path,"ai_challenger_scene_train_20170904/scene_train_images_20170904/scene_classes.csv")) as f:
            f_csv = csv.reader(f,delimiter=',')
            for row in f_csv:
                id2chi[row[0]] = row[1]
                id2eng[row[0]] = row[2]

        if part not in which:
            raise ValueError('please specify the correct part of dataset.')

        self.data = {which[i]:"{}/{}".format(path,"ai_challenger_scene_train_20170904/scene_%s_images_20170904" % which[i]) for i in range(4)}
        self.label = {which[i]:"{}/{}/train_label.txt".format(path,"ai_challenger_scene_train_20170904/scene_%s_images_20170904" % which[i]) for i in range(4)}
        self.part, self.Transform = part, Transform

    def __len__(self):

        with open(self.label[self.part]) as f:
            return len(f.readlines())

    def __getitem__(self, item):

        with open(self.label[self.part]) as f:
            lines = f.readlines()
            img_name,img_label = lines[item].split(' ')
            image= io.imread('{}/{}'.format(self.data[self.part],img_name))
        sample = {"image" : image, "label" : img_label}

        if self.Transform:
            sample = self.Transform(sample)

        return sample







