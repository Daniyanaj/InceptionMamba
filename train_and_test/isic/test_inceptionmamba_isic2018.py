#!/usr/bin/env python
# coding: utf-8

# # SA2Net - ISIC2018
# ---

# ## Import packages & functions

from __future__ import print_function, division

import os
import sys

sys.path.append("../..")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import copy
import json
import importlib
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm

# from torchmetrics import HausdorffDistance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from losses import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss

from utils import (
    show_sbs,
    load_config,
    _print,
)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode

# ## Set the seed

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
import random

random.seed(0)

# ## Load the config

CONFIG_NAME = "isic/isic2018_mamba_Exp9_1.yaml"
CONFIG_FILE_PATH = os.path.join("./configs", CONFIG_NAME)

config = load_config(CONFIG_FILE_PATH)
_print("Config:", "info_underline")
print(json.dumps(config, indent=2))
print(20 * "~-", "\n")

# ## Dataset and Dataloader

from datasets.isic import ISIC2018DatasetFast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# ------------------- params --------------------
INPUT_SIZE = config["dataset"]["input_size"]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ----------------- dataset --------------------
# preparing training dataset
tr_dataset = ISIC2018DatasetFast(
    data_dir=config["dataset"]["training"]["params"]["data_dir"],
    mode="tr",
    one_hot=True,
)
vl_dataset = ISIC2018DatasetFast(
    data_dir=config["dataset"]["training"]["params"]["data_dir"],
    mode="vl",
    one_hot=True,
)
te_dataset = ISIC2018DatasetFast(
    data_dir=config["dataset"]["training"]["params"]["data_dir"],
    mode="te",
    one_hot=True,
)

# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing
# !cat ~/deeplearning/skin/Prepare_ISIC2018.py

print(f"Length of trainig_dataset:\t{len(tr_dataset)}")
print(f"Length of validation_dataset:\t{len(vl_dataset)}")
print(f"Length of test_dataset:\t\t{len(te_dataset)}")

# prepare train dataloader
tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])

# prepare validation dataloader
vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])

# prepare test dataloader
te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

# -------------- test -----------------
# test and visualize the input data
for sample in tr_dataloader:
    img = sample["image"]
    msk = sample["mask"]
    print("\n Training")
    show_sbs(img[0], msk[0, 1])
    break

for sample in vl_dataloader:
    img = sample["image"]
    msk = sample["mask"]
    print("Validation")
    show_sbs(img[0], msk[0, 1])
    break

for sample in te_dataloader:
    img = sample["image"]
    msk = sample["mask"]
    print("Test")
    show_sbs(img[0], msk[0, 1])
    break

# ### Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {device}")

# ## Metrics

metrics = torchmetrics.MetricCollection(
    [
        torchmetrics.F1Score(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
        torchmetrics.Accuracy(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
        torchmetrics.Dice(),
        torchmetrics.Precision(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
        torchmetrics.Specificity(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
        torchmetrics.Recall(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
        # IoU
        torchmetrics.JaccardIndex(
            num_classes=config["dataset"]["number_classes"], task="multiclass"
        ),
    ],
    prefix="train_metrics/",
)

# train_metrics
train_metrics = metrics.clone(prefix="train_metrics/").to(device)

# valid_metrics
valid_metrics = metrics.clone(prefix="valid_metrics/").to(device)

# test_metrics
test_metrics = metrics.clone(prefix="test_metrics/").to(device)

def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res

# ## Define validate function

def validate(model, criterion, vl_dataloader):
    model.eval()
    with torch.no_grad():

        evaluator = valid_metrics.clone().to(device)

        losses = []
        cnt = 0.0
        for batch, batch_data in enumerate(vl_dataloader):
            imgs = batch_data["image"]
            msks = batch_data["mask"]

            cnt += msks.shape[0]

            imgs = imgs.to(device)
            msks = msks.to(device)

            preds = model(imgs)
            loss = criterion(preds, msks)
            losses.append(loss.item())

            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)

        #             _cml = f"curr_mean-loss:{np.sum(losses)/cnt:0.5f}"
        #             _bl = f"batch-loss:{losses[-1]/msks.shape[0]:0.5f}"
        #             iterator.set_description(f"Validation) batch:{batch+1:04d} -> {_cml}, {_bl}")

        # print the final results
        loss = np.sum(losses) / cnt
        metrics = evaluator.compute()

    return evaluator, loss

# ## Define train function

def train(
    model,
    device,
    tr_dataloader,
    vl_dataloader,
    config,
    criterion,
    optimizer,
    scheduler,
    save_dir="./",
    save_file_id=None,
):

    EPOCHS = tr_prms["epochs"]

    torch.cuda.empty_cache()
    model = model.to(device)

    evaluator = train_metrics.clone().to(device)

    epochs_info = []
    best_model = None
    best_result = {}
    best_vl_loss = np.Inf
    for epoch in range(EPOCHS):
        model.train()

        evaluator.reset()
        tr_iterator = tqdm(enumerate(tr_dataloader))
        tr_losses = []
        cnt = 0
        for batch, batch_data in tr_iterator:
            imgs = batch_data["image"]
            msks = batch_data["mask"]

            imgs = imgs.to(device)
            msks = msks.to(device)

            optimizer.zero_grad()
            preds4 = model(imgs)
            loss4 = criterion(preds4, msks)

            loss = loss4
            loss.backward()
            optimizer.step()

            preds = preds4

            # evaluate by metrics
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)

            cnt += imgs.shape[0]
            tr_losses.append(loss.item())

            # write details for each training batch
            _cml = f"curr_mean-loss:{np.sum(tr_losses)/cnt:0.5f}"
            _bl = f"mean_batch-loss:{tr_losses[-1]/imgs.shape[0]:0.5f}"
            tr_iterator.set_description(
                f"Training) ep:{epoch:03d}, batch:{batch+1:04d} -> {_cml}, {_bl}"
            )

        tr_loss = np.sum(tr_losses) / cnt

        # validate model
        vl_metrics, vl_loss = validate(model, criterion, vl_dataloader)
        if vl_loss < best_vl_loss:
            # find a better model
            best_model = model
            best_vl_loss = vl_loss
            best_result = {
                "tr_loss": tr_loss,
                "vl_loss": vl_loss,
                "tr_metrics": make_serializeable_metrics(evaluator.compute()),
                "vl_metrics": make_serializeable_metrics(vl_metrics.compute()),
            }

        # write the final results
        epoch_info = {
            "tr_loss": tr_loss,
            "vl_loss": vl_loss,
            "tr_metrics": make_serializeable_metrics(evaluator.compute()),
            "vl_metrics": make_serializeable_metrics(vl_metrics.compute()),
        }
        epochs_info.append(epoch_info)
        #         epoch_tqdm.set_description(f"Epoch:{epoch+1}/{EPOCHS} -> tr_loss:{tr_loss}, vl_loss:{vl_loss}")
        evaluator.reset()

        scheduler.step(vl_loss)

    # save final results
    res = {
        "id": save_file_id,
        "config": config,
        "epochs_info": epochs_info,
        "best_result": best_result,
    }
    fn = f"{save_file_id+'_' if save_file_id else ''}result.json"
    fp = os.path.join(config["model"]["save_dir"], fn)
    with open(fp, "w") as write_file:
        json.dump(res, write_file, indent=4)

    # save model's state_dict
    fn = "last_model_state_dict.pt"
    fp = os.path.join(config["model"]["save_dir"], fn)
    torch.save(model.state_dict(), fp)

    # save the best model's state_dict
    fn = "best_model_state_dict.pt"
    fp = os.path.join(config["model"]["save_dir"], fn)
    torch.save(best_model.state_dict(), fp)

    return best_model, model, res

# ## Define test function

# import torch
from scipy.spatial.distance import directed_hausdorff

def hd95(pred, target):
    """
    Compute the 95th percentile of the Hausdorff Distance between two binary masks.

    Args:
        pred (torch.Tensor): Predicted binary segmentation mask (H, W) or (D, H, W).
        target (torch.Tensor): Ground truth binary segmentation mask (H, W) or (D, H, W).

    Returns:
        float: The HD95 distance.
    """
    # Convert tensors to NumPy arrays
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # Ensure the input is binary
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    # Get the boundary points of both masks
    pred_boundary = np.argwhere(pred)
    target_boundary = np.argwhere(target)

    # Compute directed Hausdorff distances
    hd_pred_to_target = directed_hausdorff(pred_boundary, target_boundary)[0]
    hd_target_to_pred = directed_hausdorff(target_boundary, pred_boundary)[0]

    # Combine and take the 95th percentile
    hd95_value = np.percentile([hd_pred_to_target, hd_target_to_pred], 95)
    return hd95_value

from medpy import metric

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2.0 * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def hd(pred, gt):
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

def test(model, te_dataloader):
    model.eval()
    all_hd95 = []
    all_dices = []
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)
        for batch_data in tqdm(te_dataloader):
            imgs = batch_data["image"]
            msks = batch_data["mask"]

            imgs = imgs.to(device)
            msks = msks.to(device)

            preds = model(imgs)

            # evaluate by metrics
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)
            # print("shape of preds_ is: ", preds_.shape)
            # print("shape of msks_ is: ", msks_.shape)
            all_hd95.append(hd(preds_, msks_))
            all_dices.append(dice(preds_, msks_))
    return evaluator, all_hd95, all_dices

# ## Load and prepare model

# download weights

# !wget "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
# !mkdir -p ../model/vit_checkpoint/imagenet21k
# !mv R50+ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz

from models.mynet.mamba_Exp9_1 import MyNet as Net

# import models._uctransnet.Config as uct_config

# config_vit = uct_config.get_CTranS_config()

# model = Net(config_vit, **config['model']['params'])
model = Net(
    n_channels=config["model"]["params"]["n_channels"],
    n_classes=config["model"]["params"]["n_classes"],
)

torch.cuda.empty_cache()
model = model.to(device)
print(
    "Number of parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

from fvcore.nn import FlopCountAnalysis

input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor

# Calculate FLOPs
flops = FlopCountAnalysis(model, input_tensor)
total_flops = flops.total()

# Convert FLOPs to GFLOPs
gflops = total_flops / 1e9

print(f"Total FLOPs: {total_flops}")
print(f"GFLOPs: {gflops:.2f}")
os.makedirs(config["model"]["save_dir"], exist_ok=True)
model_path = f"{config['model']['save_dir']}/model_state_dict.pt"

if config["model"]["load_weights"]:
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained weights...")

# criterion_dice = DiceLoss()
criterion_dice = DiceLossWithLogtis()
# criterion_ce  = BCELoss()
criterion_ce = CrossEntropyLoss()

def criterion(preds, masks):
    c_dice = criterion_dice(preds, masks)
    c_ce = criterion_ce(preds, masks)
    return 0.5 * c_dice + 0.5 * c_ce

tr_prms = config["training"]
optimizer = globals()[tr_prms["optimizer"]["name"]](
    model.parameters(), **tr_prms["optimizer"]["params"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", **tr_prms["scheduler"]
)

# best_model = Net(config_vit, **config['model']['params'])
# best_model = Net(config_vit, **config['model']['params'])
best_model = Net(
    n_channels=config["model"]["params"]["n_channels"],
    n_classes=config["model"]["params"]["n_classes"],
)

torch.cuda.empty_cache()
best_model = best_model.to(device)

fn = "best_model_state_dict.pt"
os.makedirs(config["model"]["save_dir"], exist_ok=True)
model_path = f"{config['model']['save_dir']}/{fn}"

best_model.load_state_dict(torch.load(model_path))
print("Loaded best model weights from ...", model_path)

# ## Evaluation

te_metrics, all_hd95s, all_dices = test(best_model, te_dataloader)
te_metrics.compute()

print("**********************    Scores are ****************")
print("*****************************************************")
print("*****************************************************")
print(te_metrics.compute())
print("*****************************************************")
print("*****************************************************")

mean_hd95 = np.mean(all_hd95s)
std_hd95 = np.std(all_hd95s)
print("**********************    HD95 scores are ****************")
print("*****************************************************")
print("*****************************************************")
print(f"Mean HD95 Score: {mean_hd95:.2f} ± {std_hd95:.2f}")

mean_dice = np.mean(all_dices)
std_dice = np.std(all_dices)
print("**********************    all_dices scores are ****************")
print("*****************************************************")
print("*****************************************************")
print(f"Mean Dice Score: {mean_dice:.2f} ± {std_dice:.2f}")
# ## Plot graphs

result_file_path = f"{config['model']['save_dir']}/result.json"
with open(result_file_path, "r") as f:
    results = json.loads("".join(f.readlines()))
epochs_info = results["epochs_info"]

tr_losses = [d["tr_loss"] for d in epochs_info]
vl_losses = [d["vl_loss"] for d in epochs_info]
tr_dice = [d["tr_metrics"]["train_metrics/Dice"] for d in epochs_info]
vl_dice = [d["vl_metrics"]["valid_metrics/Dice"] for d in epochs_info]
tr_js = [d["tr_metrics"]["train_metrics/MulticlassJaccardIndex"] for d in epochs_info]
vl_js = [d["vl_metrics"]["valid_metrics/MulticlassJaccardIndex"] for d in epochs_info]
tr_acc = [d["tr_metrics"]["train_metrics/MulticlassAccuracy"] for d in epochs_info]
vl_acc = [d["vl_metrics"]["valid_metrics/MulticlassAccuracy"] for d in epochs_info]

_, axs = plt.subplots(1, 4, figsize=[16, 3])

axs[0].set_title("Loss")
axs[0].plot(tr_losses, "r-", label="train loss")
axs[0].plot(vl_losses, "b-", label="validatiton loss")
axs[0].legend()

axs[1].set_title("Dice score")
axs[1].plot(tr_dice, "r-", label="train dice")
axs[1].plot(vl_dice, "b-", label="validation dice")
axs[1].legend()

axs[2].set_title("Jaccard Similarity")
axs[2].plot(tr_js, "r-", label="train JaccardIndex")
axs[2].plot(vl_js, "b-", label="validatiton JaccardIndex")
axs[2].legend()

axs[3].set_title("Accuracy")
axs[3].plot(tr_acc, "r-", label="train Accuracy")
axs[3].plot(vl_acc, "b-", label="validation Accuracy")
axs[3].legend()

plt.show()

epochs_info

# ## Save images
