import os
import csv
import math
import json
import types
import torch
import random
import numbers
import collections
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import Compose

def pil_loader(path):
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
            return {'image':sample['image'].transpose(Image.FLIP_LEFT_RIGHT),'label':sample['label'],'idx':sample['idx']}
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
                return {'image':sample['image'].resize((ow, oh), self.interpolation),'label':sample['label'],'idx' : sample['idx']}
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return {'image':sample['image'].resize((ow, oh), self.interpolation),'label':sample['label'],'idx' : sample['idx']}
        else:
            return {'image' : sample['image'].resize(self.size, self.interpolation),'label':sample['label'],'idx' : sample['idx']}

class RandomScaleCrop(object):

    def __init__(self,interpolation=Image.BILINEAR):
        self.size = random.choice([256,384,512])
        self.cropSize = {"256":224,"384":336,"512":448}
        self.interpolation = interpolation

    def __call__(self, sample):

        w, h = sample['image'].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return sample
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            img = sample['image'].resize((ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            img = sample['image'].resize((ow, oh), self.interpolation)

        for attempt in range(100):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return {'image':img.resize((224, 224), self.interpolation),'label':sample['label'],'idx':sample['idx']}

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
        return {'image':sample['image'].crop((x1,y1,x1+tw,y1+th)),'label':sample['label'],'idx':sample['idx']}

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

                return {'image':sample['image'].resize((self.size,self.size),self.interpolation),'label':sample['label'],'idx':sample['idx']}

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return {'image':crop(scale(sample['image'])),'label':sample['label'],'idx':sample['idx']}

class supervised_Crop(object):

    """Crop the given PIL.Image to given crop size

        Args:
            size: crop size
            interpolation: Default: PIL.Image.BILINEAR
        """

    def __init__(self,crop, path, interpolation=Image.BILINEAR):
        if not isinstance(crop,tuple):
            raise ValueError('specify the crop size(tuple)')
        self.crop = crop # 224, 336, 448 images already been scaled to (256,256),(384,384),(512,512)
        self.interpolation = interpolation
        self.store = np.load(os.path.join(path,"crop_probs.npz"))
        self.size = {'224':(256,256),'336':(384,384),'448':(512,512)}

    def __call__(self,sample):

        assert isinstance(sample['idx'],int)
        if sample['idx'] in self.store['index']:
            coordinates = self.store['coordinates'][self.store['index'].index(sample['idx'])] # index for generated heatmap images, logical error
            coordinate = random.randint(0,len(coordinates))
            r = coordinate // (self.size[self.crop][0] - self.crop + 1)
            c = coordinate % (self.size[self.crop][0] - self.crop + 1)
            sample['image'].crop((r,c,r+self.crop,c+self.crop))
            return {'image':sample['image'],'label':sample['label'],'idx':sample['idx']}
        else:
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
                    assert (sample['image'].size == (w, h))

                    return {'image': sample['image'].resize((self.crop, self.crop), self.interpolation),
                            'label': sample['label'],
                            'idx':sample['idx']}
        # Fallback
        scale = Scale(self.crop, interpolation=self.interpolation)
        crop = CenterCrop(self.crop)
        return {'image': crop(scale(sample['image'])), 'label': sample['label'],'idx':sample['idx']}

class FiveCrop(object):
    """Crop the given PIL.Image into four corners and the central crop.abs

       Note: this transform returns a tuple of images and there may be a mismatch in the number of
       inputs and targets your `Dataset` returns.

       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, sample):

        w, h = sample['image'].size
        crop_h, crop_w = self.size
        if crop_w > w or crop_h > h:
            raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size,
                                                                                          (h, w)))
        tl = sample['image'].crop((0, 0, crop_w, crop_h))
        tr = sample['image'].crop((w - crop_w, 0, w, crop_h))
        bl = sample['image'].crop((0, h - crop_h, crop_w, h))
        br = sample['image'].crop((w - crop_w, h - crop_h, w, h))
        tmp = CenterCrop((crop_h, crop_w))(sample)
        center = tmp['image']
        return {"image":(tl, tr, bl, br, center),"label":sample['label'],"idx":sample['idx']}


class TenCrop(object):
    """Crop the given PIL.Image into four corners and the central crop plus the flipped version of these
       (horizontal flipping is used by default)

       Note: this transform returns a tuple of images and there may be a mismatch in the number of
       inputs and targets your `Dataset` returns.

       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
                made.
           vflip bool: Use vertical flipping instead of horizontal
    """

    def __init__(self, size, vflip=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vflip = vflip

    def __call__(self, sample):
        five_crop = FiveCrop(self.size)
        first_five = five_crop(sample)['image']
        if self.vflip:
            img = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
        else:
            img = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)

        second_five = five_crop({'image':img,'label':sample['label'],'idx':sample['idx']})['image']
        return {'image':first_five + second_five,'label':sample['label'],'idx':sample['idx']}

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self,eval=False):
        self.eval = eval

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        if self.eval:
            assert isinstance(sample['image'],tuple)
            assert len(sample['image']) == 10
            imgs = list()
            for i in range(10):
                # handle PIL Image
                if sample['image'][i].mode == 'I':
                    img = torch.from_numpy(np.array(sample['image'][i], np.int32, copy=False))
                elif sample['image'][i].mode == 'I;16':
                    img = torch.from_numpy(np.array(sample['image'][i], np.int16, copy=False))
                else:
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(sample['image'][i].tobytes()))
                # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
                if sample['image'][i].mode == 'YCbCr':
                    nchannel = 3
                elif sample['image'][i].mode == 'I;16':
                    nchannel = 1
                else:
                    nchannel = len(sample['image'][i].mode)
                img = img.view(sample['image'][i].size[1], sample['image'][i].size[0], nchannel)
                # put it from HWC to CHW format
                # yikes, this transpose takes 80% of the loading time/CPU
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    imgs.append(img.float().div(255))
                else:
                    imgs.append(img)
            return {'image':imgs,'label':sample['label'],'idx':sample['idx']}

        if isinstance(sample['image'], np.ndarray):
            # handle numpy array
            img = torch.from_numpy(sample['image'].transpose((2, 0, 1)))
            # backward compatibility
            return {'image':img.float().div(255),'label':sample['label']}

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
            return {'image':img.float().div(255),'label':sample['label'], 'idx':sample['idx']}
        else:
            return {'image':img,'label':sample['label'],'idx':sample['idx']}

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

    def __init__(self, mean, std, eval=False):
        self.mean = mean
        self.std = std
        self.eval = eval

    def __call__(self, sample):
        """
        Args:
            sample['image'] (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient

        if self.eval:
            assert isinstance(sample['image'],list)
            assert len(sample['image']) == 10
            for i in range(10):
                for t,m,s in zip(sample['image'][i],self.mean,self.std):
                    t.sub_(m).div_(s)
            return {'image':sample['image'],'label':sample['label'],'idx':sample['idx']}
        else:
            for t, m, s in zip(sample['image'], self.mean, self.std):
                t.sub_(m).div_(s)
            return {'image':sample['image'],'label':sample['label'],'idx':sample['idx']}

class pcaJittering(object):
    """
    implements fancy PCA Jittering of AlexNet
    """
    def __init__(self,eig):
        self.eigValue = eig[0]
        self.eigVector = eig[1]

    def __call__(self,sample):
        alpha = np.array([random.gauss(0, 0.1) for i in range(3)],dtype="float32")
        addQuantity = np.dot(self.eigVector,alpha * self.eigValue)
        addImg = sample['image'].numpy() + addQuantity.reshape((3,1,1))
        return {'image':torch.from_numpy(addImg),'label':sample['label'],'idx':sample['idx']}

def adjust_brightness(sample, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """

    enhancer = ImageEnhance.Brightness(sample['image'])
    img = enhancer.enhance(brightness_factor)
    return {'image':img,'label':sample['label'],'idx':sample['idx']}


def adjust_contrast(sample, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        sample['image'] (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """

    enhancer = ImageEnhance.Contrast(sample['image'])
    img = enhancer.enhance(contrast_factor)
    return {'image':img,'label':sample['label'],'idx':sample['idx']}


def adjust_saturation(sample, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        sample['image'] (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """

    enhancer = ImageEnhance.Color(sample['image'])
    img = enhancer.enhance(saturation_factor)
    return {'image':img,'label':sample['label'],'idx':sample['idx']}


def adjust_hue(sample, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = sample['image'].mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return {'image':sample['image'],'label':sample['label'],'idx':sample['idx']}

    h, s, v = sample['image'].convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return {'image':img,'label':sample['label'],'idx':sample['idx']}

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            sample['image'] (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(sample)

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
            with open(os.path.join(path,sub_path[part],"train_label.txt"),'w') as f_label:
                for i in range(len(f)):
                    dict = f[i]
                    f_label.write("{} {}\n".format(dict['image_id'],dict['label_id']))
        else:
            raise ValueError('specify the root path!')
        self.read = os.path.join(path, sub_path[part], "%s_label.txt" % part)
        self.image,self.label = list(),list()

        # read txt file, store full image/label path in self instance
        with open(self.read) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                img_name, label_index = lines[i].split(' ')
                self.image.append(os.path.join(path, sub_path[part], img_path[part], img_name))
                self.label.append(int(label_index))

        self.part, self.Transform, self.id2chi, self.id2eng = part ,Transform, id2chi, id2eng

    def __len__(self):

        with open(self.read,'r') as f:
            length = len(f.readlines())
        return length

    def __getitem__(self,item):

        image = pil_loader(self.image[item])
        sample = {"image":image,"label":self.label[item],"idx":item}

        if self.Transform:
            tsfm_sample = self.Transform(sample)
            return tsfm_sample
        else:
            return sample

class AIC_scene_test(Dataset):
    global id2chi, id2eng, sub_path, img_path
    id2chi, id2eng = dict(), dict()
    sub_path = {"testA": "ai_challenger_scene_test_a_20170922"}
    img_path = {"testA": "scene_test_a_images_20170922"}

    def __init__(self,part="testA",path=None,Transform=None):


        if path is not None:
            with open(os.path.join(path,sub_path[part],"scene_classes.csv"), 'r') as f:
                f_csv = csv.reader(f, delimiter=',')
                for row in f_csv:
                    id2chi[row[0]] = row[1]
                    id2eng[row[0]] = row[2]
        else:
            raise ValueError('specify the root path!')

        leaf_path = os.path.join(path,sub_path[part],img_path[part])
        file_path_tmp = os.path.join(path,sub_path[part],"{}_tmp.txt".format(part))
        file_path = os.path.join(path,sub_path[part],"{}.txt".format(part))
        os.system("leaf_path={}".format(leaf_path))
        os.system("file={}".format(file_path_tmp))
        os.system("find {} -name *.jpg > {}".format(leaf_path,file_path_tmp))
        data = list()
        with open(file_path_tmp) as f:
            lines=f.readlines()
            for i in range(len(lines)):
                data.append(lines[i].replace("{}/".format(os.path.join(path,sub_path[part],img_path[part])),""))
        with open(file_path,'w') as f:
            f.writelines(data)

        self.path,self.part,self.image,self.image_name,self.Transform = path,part,list(),list(),Transform
        self.read = os.path.join(path,sub_path[part],"{}.txt".format(part))
        with open(self.read) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                img_name = lines[i]
                self.image_name.append(img_name)
                self.image.append(os.path.join(path,sub_path[part],img_path[part],img_name))

    def __len__(self):

        with open(self.read) as f:
            lines = f.readlines()
            length = len(lines)

        return length

    def __getitem__(self,item):

        image = pil_loader(self.image[item].rstrip())
        sample = {"image": image, "label" : self.image_name[item],"idx": item}

        if self.Transform:
            tsfm_sample = self.Transform(sample)
            return tsfm_sample

        return sample



