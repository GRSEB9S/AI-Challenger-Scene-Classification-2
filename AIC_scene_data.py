from torch.utils.data import Dataset
import csv
import random
from PIL import Image
import os
import math
import collections
import numpy as np
import torch
import json
try:
    import accimage
except ImportError:
    accimage = None
import numbers

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return {'image':sample['image'].transpose(Image.FLIP_LEFT_RIGHT),'label':sample['label']}
        return sample

class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """

        if isinstance(self.size, int):
            w, h = sample['image'].size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return sample
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return {'image':sample['image'].resize((ow, oh), self.interpolation),'label':sample['label']}
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return {'image':sample['image'].resize((ow, oh), self.interpolation),'label':sample['label']}
        else:
            return {'image' : sample['image'].resize(self.size, self.interpolation),'label' : sample['label']}

class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = sample['image'].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return {'image' : sample['image'].crop((x1, y1, x1 + tw, y1 + th)),'label' : sample['label']}

class RandomSizedCrop(object):

    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        for attempt in range(100):
            area = sample['image'].size[0] * sample['image'].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= sample['image'].size[0] and h <= sample['image'].size[1]:
                x1 = random.randint(0, sample['image'].size[0] - w)
                y1 = random.randint(0, sample['image'].size[1] - h)

                sample['image'] = sample['image'].crop((x1, y1, x1 + w, y1 + h))
                assert(sample['image'].size == (w, h))

                return {'image':sample['image'].resize((self.size, self.size), self.interpolation),'label':sample['label']}

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return {'image' : crop(scale(sample['image'])),'label' : sample['label']}

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(sample['image'], np.ndarray):
            # handle numpy array
            img = torch.from_numpy(sample['image'].transpose((2, 0, 1)))
            # backward compatibility
            return {'image':img.float().div(255),'label':sample['label']}

        if accimage is not None and isinstance(sample['image'], accimage.Image):
            nppic = np.zeros([sample['image'].channels, sample['image'].height, sample['image'].width], dtype=np.float32)
            sample['image'].copyto(nppic)
            return {'image':torch.from_numpy(nppic),'label':sample['label']}

        # handle PIL Image
        if sample['image'].mode == 'I':
            img = torch.from_numpy(np.array(sample['image'], np.int32, copy=False))
        elif sample['image'].mode == 'I;16':
            img = torch.from_numpy(np.array(sample['image'], np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(sample['image'].tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if sample['image'].mode == 'YCbCr':
            nchannel = 3
        elif sample['image'].mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(sample['image'].mode)
        img = img.view(sample['image'].size[1], sample['image'].size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return {'image':img.float().div(255),'label':sample['label']}
        else:
            return {'image':img,'label':sample['label']}

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            sample['image'] (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(sample['image'], self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image':sample['image'],'label':sample['label']}


class AIC_scene(Dataset):

    global id2chi, id2eng
    id2chi, id2eng = dict(), dict()

    def __init__(self,part="train",path=None,Transform=None):

        sub_path = {"train": "ai_challenger_scene_train_20170904",
                    "val": "ai_challenger_scene_validation_20170908"}
        img_path = {"train" : "scene_train_images_20170904",
                    "val" : "scene_validation_images_20170908"}
        json_name = {"train":"scene_train_annotations_20170904.json",
                     "val":"scene_validation_annotations_20170908.json"}

        # convert json file to txt file
        if path is not None:
            with open(os.path.join(path,sub_path[part],"scene_classes.csv"), 'r') as f:
                f_csv = csv.reader(f, delimiter=',')
                for row in f_csv:
                    id2chi[row[0]] = row[1]
                    id2eng[row[0]] = row[2]

            f = json.load(open(os.path.join(path,sub_path[part],json_name[part])))
            with open(os.path.join(path,sub_path[part],"%s_label.txt" % part),"w") as f_label:
                for i in range(len(f)):
                    dict = f[i]
                    f_label.write("{} {}\n".format(dict['image_id'],dict['label_id']))
        else:
            raise ValueError('specify the root path!')

        self.read = os.path.join(path, sub_path[part], "%s_label.txt" % part)
        self.image, self.label = list(), list()

        # read txt file, store full image/label path in self instance
        with open(self.read) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                img_name,label_index = lines[i].split(' ')
                self.image.append(os.path.join(path,sub_path[part],img_path[part],img_name))
                self.label.append(int(label_index))

        self.part, self.path, self.Transform, self.id2chi, self.id2eng = part, path, Transform, id2chi, id2eng

    def __len__(self):

        with open(self.read, 'r') as f:
            lenth = len(f.readlines())
        return lenth

    def __getitem__(self, item):

        image = pil_loader(self.image[item])
        sample = {"image": image, "label": self.label[item]}

        if self.Transform:
            tsfm_sample = self.Transform(sample)
            return tsfm_sample

        return sample

class places365std_AIC(Dataset):

    # add images with common classes to AIC_scene_train dataset
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

class LSUN(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass



