"""
Pretrain on tumor cifar with a Masked AutoEncoder
Kaiming He et al. Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/pdf/2111.06377.pdf)
"""
import os
import sys
import glob
import numpy as np
import math
import pandas as pd
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import seed_everything, load_config, EarlyStopper
from datasets import TumorCifarDataset
from ViT import SimpleViT
from MAE import MAE
from TimeDistanceViT import TimeDistanceViT
from train import pretrain_MAE
# from test import test
from scheduler import ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineSchedule

model_classes = {
    "ViT": SimpleViT,
    "TimeDistanceViT": TimeDistanceViT
}
schedules = {
    "Constant": ConstantLRSchedule,
    "WarmupConstant": WarmupConstantSchedule,
    "Linear": WarmupLinearSchedule,
    "Cosine": WarmupCosineSchedule,
}

def run_pretrain(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    train_dir = config["train_dir"]
    train_seq_dir = [os.path.join(train_dir, f"T{t}") for t in range(5)]
    img_names = [os.path.basename(i) for i in glob.glob(os.path.join(train_dir, "T0", "*.png"))]
    label_df = pd.read_csv(config["train_label"])
    log_dir = os.path.join(config["root_dir"], "logs", config_id)
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id)
    model_dir = os.path.join(config["root_dir"], "models", config_id)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(config["root_dir"], "models", config_id)).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training/Validation split
    val_size = int(np.floor(config["val_fraction"] * len(img_names)))
    np.random.shuffle(img_names)
    val_names, train_names = img_names[:val_size], img_names[val_size:]

    # Datasets and loaders
    time_range = (0, 4)
    train_img_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=(0,90)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_img_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = TumorCifarDataset(train_dir, train_names, label_df, time_range, img_transform=train_img_transforms)
    val_dataset = TumorCifarDataset(train_dir, val_names, label_df, time_range, img_transform=val_img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), num_workers=2)

    model_class = model_classes[config["model_class"]]
    vit = model_class(
        image_size=32,
        patch_size=config["patch_size"],
        num_classes=1000,
        dim=config["embedding_dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        qkv_bias=config["qkv_bias"],
        time_embedding=config["time_emb"],
        pos_embedding=config["pos_emb"],
        dim_head=config["dim_head"],
    ).to(device)
    mae = MAE(
        encoder=vit,
        encoder_class=config["model_class"],
        masking_ratio=config["masking_ratio"],
        decoder_dim=config["decoder_dim"],
        decoder_depth=config["decoder_depth"],
    ).to(device)

    # optimizer
    optimizer = optim.AdamW(mae.parameters(), lr=config["lr"], betas=(0.9, 0.95))

    # resume training from checkpoint if indicated
    if config["checkpoint"]:
        print(f"Resuming training of {config_id} from {config['checkpoint']}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, config['checkpoint']))
        mae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_metric = checkpoint['best_metric']
        start_epoch = checkpoint['epoch'] + 1
        last_global_step = checkpoint['global_step']
    else:
        best_metric = 1e5
        start_epoch = 0
        last_global_step = 0
    # epochs
    epoch_range = (start_epoch, config["epochs"])

    # scheduler
    n_batches = math.ceil(len(train_dataset)/int(config["batch_size"]))
    print(f"Total steps: {config['epochs']*n_batches}")
    if config["schedule"]=="Constant":
        scheduler = schedules[config["schedule"]](optimizer, last_epoch=start_epoch*n_batches-1)
    elif config["schedule"]=="WarmupConstant":
        scheduler = schedules[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                 last_epoch=start_epoch*n_batches-1)
    else:
        scheduler = schedules[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                 t_total=config["epochs"]*n_batches, last_epoch=start_epoch*n_batches-1)

    # Termination criteria
    stopper = EarlyStopper(config["stop_agg"], config["stop_delta"])
    # writer
    writer = SummaryWriter(log_dir=log_dir)

    pretrain_MAE(config, config_id, vit, mae, epoch_range, last_global_step, train_loader, val_loader, device,
          best_metric, optimizer, scheduler, stopper, writer, model_dir, checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    config_dir = "/home/local/VANDERBILT/litz/github/MASILab/time-distance-transformer/configs"
    config = load_config(config_dir, args.config)
    if args.train:
        run_pretrain(config, args.config)
    # if args.test:
    #     run_test(config, args.config)
