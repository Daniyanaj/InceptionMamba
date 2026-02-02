import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F

import random

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms

def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [
        torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0) for img in labels
    ]

    imgs = [
        TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for img in imgs
    ]
    return imgs, labels

class SegPC2021Dataset_train(Dataset):
    def __init__(
        self,
        mode,  # 'tr'-> train, 'vl' -> validation, 'te' -> test
        input_size=224,
        scale=2.5,
        data_dir=None,
        dataset_dir=None,
        one_hot=True,
        force_rebuild=False,
        img_transform=None,
        msk_transform=None,
    ):
        # pre-set variables
        self.data_dir = (
            data_dir
            if data_dir
            else "/nvme-data/Medical/Segmentation_UNet/datasets/SegPC/np"
        )
        self.dataset_dir = (
            dataset_dir
            if dataset_dir
            else "/nvme-data/Medical/Segmentation_UNet/datasets/SegPC/TCIA_SegPC_dataset/"
        )
        self.mode = mode
        # input parameters
        self.input_size = input_size
        self.scale = scale
        self.one_hot = one_hot

        self.img_transform = DataAugmentation(
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
        )

        # loading data
        self.load_dataset(force_rebuild=force_rebuild)

    def load_dataset(self, force_rebuild):
        INPUT_SIZE = self.input_size
        ADD = self.data_dir

        #         build_segpc_dataset(
        #             input_size = self.input_size,
        #             scale = self.scale,
        #             data_dir = self.data_dir,
        #             dataset_dir = self.dataset_dir,
        #             mode = self.mode,
        #             force_rebuild = force_rebuild,
        #         )

        print(f"loading X_{self.mode}...")
        self.X = np.load(
            f"{ADD}/cyts_{self.mode}_{self.input_size}x{self.input_size}_s{self.scale}_X.npy"
        )
        print(f"loading Y_{self.mode}...")
        self.Y = np.load(
            f"{ADD}/cyts_{self.mode}_{self.input_size}x{self.input_size}_s{self.scale}_Y.npy"
        )
        print("finished.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        msk = self.Y[idx]
        msk = np.where(msk < 0.5, 0, 1)
        img, msk = self.img_transform.transform(img, msk)
        img = (img - img.min()) / (img.max() - img.min())
        msk = (msk - msk.min()) / (msk.max() - msk.min())

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        sample = {"image": img, "mask": msk, "id": idx}
        return sample

class DataAugmentation:
    def __init__(
        self,
        with_random_hflip=False,
        with_random_vflip=False,
        with_random_rot=False,
        with_random_crop=False,
        with_scale_random_crop=False,
        with_random_blur=False,
        random_color_tf=False,
    ):

        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = TF.to_pil_image(imgs)
        labels = TF.to_pil_image(labels.astype(np.float32))

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = TF.hflip(imgs)
            labels = TF.hflip(labels)

        if self.with_random_vflip and random.random() > 0.5:

            imgs = TF.vflip(imgs)
            labels = TF.vflip(labels)

        if to_tensor:
            # to tensor
            imgs = TF.to_tensor(imgs)
            labels = TF.to_tensor(labels)

        return imgs, labels

def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0] : box[1], box[2] : box[3]] = img[box[4] : box[5], box[6] : box[7]]

    return Image.fromarray(cont)

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return (
        cont_top,
        cont_top + ch,
        cont_left,
        cont_left + cw,
        img_top,
        img_top + ch,
        img_left,
        img_left + cw,
    )

def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)

def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
