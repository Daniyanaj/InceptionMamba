#!/usr/bin/env python
# coding: utf-8

# # Prepare ISIC dataset
# The result of this preparation is used in [ISIC2018DatasetFast](https://github.com/NITR098/Awesome-U-Net/blob/72a672728376cc08c62df66df6ed8b90b329667f/datasets/isic.py#L85).

import glob
from tqdm import tqdm
import numpy as np
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import os
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transform

# Parameters
H_INPUT_SIZE = 224
W_INPUT_SIZE = 224
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

# preparing input info.
data_prefix = "ISIC_"
target_postfix = "_segmentation"
target_fex = "png"
input_fex = "jpg"
data_dir = "/dccstor/urban/mustansar/codes/med/codes/sasanet/SA2-Net-main/sa2net/SA2-Net-main/UNet_Awesome/isic2017/train"
imgs_dir = os.path.join(data_dir, "ISIC-2017_Training_Data")
msks_dir = os.path.join(data_dir, "ISIC-2017_Training_Part1_GroundTruth")

img_dirs = glob.glob(f"{imgs_dir}/*.{input_fex}")
data_ids = [d.split(data_prefix)[1].split(f".{input_fex}")[0] for d in img_dirs]

def get_img_by_id(id):
    img_dir = os.path.join(imgs_dir, f"{data_prefix}{id}.{input_fex}")
    # img = read_image(img_dir, ImageReadMode.RGB)
    img = PIL.Image.open(img_dir)
    img = transform.to_tensor(img)
    return img

def get_msk_by_id(id):
    msk_dir = os.path.join(msks_dir, f"{data_prefix}{id}{target_postfix}.{target_fex}")
    # msk = read_image(msk_dir, ImageReadMode.GRAY)
    msk = PIL.Image.open(msk_dir).convert("L")
    msk = transform.to_tensor(msk)
    return msk

# gathering images
imgs = []
msks = []
for data_id in tqdm(data_ids):
    img = get_img_by_id(data_id)
    msk = get_msk_by_id(data_id)

    if img_transform:
        img = img_transform(img)
        # img = (img - img.min()) / (img.max() - img.min())
    if msk_transform:
        msk = msk_transform(msk)
        # msk = (msk - msk.min()) / (msk.max() - msk.min())

    img = img.numpy()
    msk = msk.numpy()

    imgs.append(img)
    msks.append(msk)

X = np.array(imgs)
Y = np.array(msks)

np.save(f"{data_dir}/np_224_224/X_tr_224x224", X)
np.save(f"{data_dir}/np_224_224/Y_tr_224x224", Y)
# /dccstor/urban/mustansar/codes/med/codes/sasanet/SA2-Net-main/sa2net/SA2-Net-main/UNet_Awesome/isic2017/
# /dccstor/urban/mustansar/codes/med/codes/sasanet/SA2-Net-main/sa2net/SA2-Net-main/UNet_Awesome/isic2017/np

# get_ipython().system('ls /path/to/datasets/ISIC2018/np')

# ### Visualization

# import sys
# sys.path.append('..')
# from utils import show_sbs

# show_sbs(X[4], Y[4])
