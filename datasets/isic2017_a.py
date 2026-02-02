#!/usr/bin/env python
# coding: utf-8

# ## [ISIC Challenge (2016-2020)](https://challenge.isic-archive.com/)
# ---
#
# ### [Data 2018](https://challenge.isic-archive.com/data/)
#
# The input data are dermoscopic lesion images in JPEG format.
#
# All lesion images are named using the scheme `ISIC_<image_id>.jpg`, where `<image_id>` is a 7-digit unique identifier. EXIF tags in the images have been removed; any remaining EXIF tags should not be relied upon to provide accurate metadata.
#
# The lesion images were acquired with a variety of dermatoscope types, from all anatomic sites (excluding mucosa and nails), from a historical sample of patients presented for skin cancer screening, from several different institutions. Every lesion image contains exactly one primary lesion; other fiducial markers, smaller secondary lesions, or other pigmented regions may be neglected.
#
# The distribution of disease states represent a modified "real world" setting whereby there are more benign lesions than malignant lesions, but an over-representation of malignancies.

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

# # ----------------- transform ------------------
H_INPUT_SIZE = 1024
W_INPUT_SIZE = 768
# transform for image
img_transform = transforms.Compose(
    [
        transforms.Resize(
            size=[H_INPUT_SIZE, W_INPUT_SIZE],
            interpolation=transforms.functional.InterpolationMode.BILINEAR,
        ),
    ]
)
# transform for mask
msk_transform = transforms.Compose(
    [
        transforms.Resize(
            size=[H_INPUT_SIZE, W_INPUT_SIZE],
            interpolation=transforms.functional.InterpolationMode.NEAREST,
        ),
    ]
)

class ISIC2018Dataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        mode="train",
        one_hot=True,
        img_transform=img_transform,
        msk_transform=msk_transform,
    ):
        # pre-set variables
        self.data_prefix = "ISIC_"
        self.target_postfix = "_segmentation"
        self.target_fex = "png"
        self.input_fex = "jpg"
        self.data_dir = data_dir if data_dir else "/path/to/datasets/ISIC2017"
        self.imgs_dir = os.path.join(self.data_dir, "ISIC2018_Task1-2_Training_Input")
        self.msks_dir = os.path.join(
            self.data_dir, "ISIC2018_Task1_Training_GroundTruth"
        )

        # input parameters
        self.img_dirs = glob.glob(f"{self.imgs_dir}/*.{self.input_fex}")
        self.data_ids = [
            d.split(self.data_prefix)[1].split(f".{self.input_fex}")[0]
            for d in self.img_dirs
        ]
        self.one_hot = one_hot
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode
        if mode == "tr":
            self.imgs = self.data_ids[0:1815]
            self.msks = self.img_dirs[0:1815]
        elif mode == "vl":
            self.imgs = self.data_ids[1815 : 1815 + 259]
            self.msks = self.img_dirs[1815 : 1815 + 259]
        elif mode == "te":
            self.imgs = self.data_ids[1815 + 259 : 2594]
            self.msks = self.img_dirs[1815 + 259 : 2594]

    def get_img_by_id(self, id):
        img_dir = os.path.join(
            self.imgs_dir, f"{self.data_prefix}{id}.{self.input_fex}"
        )
        img = read_image(img_dir, ImageReadMode.RGB)
        return img

    def get_msk_by_id(self, id):
        msk_dir = os.path.join(
            self.msks_dir,
            f"{self.data_prefix}{id}{self.target_postfix}.{self.target_fex}",
        )
        msk = read_image(msk_dir, ImageReadMode.GRAY)
        return msk

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = self.imgs[idx]
        img = self.get_img_by_id(data_id)
        msk = self.get_msk_by_id(data_id)

        if self.img_transform:
            img = self.img_transform(img)
            img = (img - img.min()) / (img.max() - img.min())
        if self.msk_transform:
            msk = self.msk_transform(msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min())

        imgs = TF.to_tensor(imgs)
        labels = TF.to_tensor(labels)

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        sample = {"image": img, "mask": msk, "id": data_id}
        return sample

class ISIC2017DatasetFast(Dataset):
    def __init__(
        self, mode, data_dir=None, one_hot=True, img_transform=None, msk_transform=None
    ):
        # pre-set variables
        self.data_dir = (
            data_dir
            if data_dir
            else "/dccstor/urban/mustansar/codes/med/codes/sasanet/SA2-Net-main/sa2net/SA2-Net-main/UNet_Awesome/isic2017/np"
        )

        # input parameters
        self.one_hot = one_hot
        print("We are in ISIC2017DatasetFast at line 147")
        X = np.load(f"{self.data_dir}/X_tr_224x224.npy")
        Y = np.load(f"{self.data_dir}/Y_tr_224x224.npy")
        print("We are in ISIC2017DatasetFast at line 150")
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        self.img_transform = DataAugmentation(
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
        )
        self.imgs = X
        self.msks = Y

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]
        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        # labels = TF.to_pil_image(labels.astype(np.float32))
        msk = TF.to_pil_image(msk)

        img = np.array(img).astype(np.float32)
        msk = np.array(msk).astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min())
        msk = (msk - msk.min()) / (msk.max() - msk.min())
        img = TF.to_tensor(img)
        msk = TF.to_tensor(msk)

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        sample = {"image": img, "mask": msk, "id": data_id}
        return sample

class ISIC2017DatasetFast_training(Dataset):
    def __init__(
        self, mode, data_dir=None, one_hot=True, img_transform=None, msk_transform=None
    ):
        # pre-set variables
        self.data_dir = (
            data_dir
            if data_dir
            else "/dccstor/urban/mustansar/codes/med/codes/sasanet/SA2-Net-main/sa2net/SA2-Net-main/UNet_Awesome/isic2017//np"
        )

        # input parameters
        self.one_hot = one_hot
        print("We are in ISIC2017DatasetFast_training at line 201")
        X = np.load(f"{self.data_dir}/X_tr_224x224.npy")
        Y = np.load(f"{self.data_dir}/Y_tr_224x224.npy")
        print("We are in ISIC2017DatasetFast_training at line 205")
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        self.img_transform = DataAugmentation(
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
        )
        self.imgs = X
        self.msks = Y

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]
        img, msk = self.img_transform.transform(img, msk)

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        sample = {"image": img, "mask": msk, "id": data_id}
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
        # labels = TF.to_pil_image(labels.astype(np.float32))
        labels = TF.to_pil_image(labels)

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = TF.hflip(imgs)
            labels = TF.hflip(labels)

        if self.with_random_vflip and random.random() > 0.5:

            imgs = TF.vflip(imgs)
            labels = TF.vflip(labels)

        imgs = np.array(imgs).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
        labels = (labels - labels.min()) / (labels.max() - labels.min())

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
