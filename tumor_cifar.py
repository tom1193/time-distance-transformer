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
from TimeDistanceViT import TimeDistanceViT
from train import train
from test import test
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

def run_train(config, config_id):
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
    Path(model_dir).mkdir(parents=True, exist_ok=True)
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

    # model
    model_class = model_classes[config["model_class"]]
    model = model_class(
        image_size=32,
        patch_size=config["patch_size"],
        num_classes=2,
        dim=config["embedding_dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        qkv_bias=config["qkv_bias"],
        time_embedding=config["time_emb"],
        pos_embedding=config["pos_emb"],
        dim_head=config["dim_head"],
    ).to(device)

    # loss function
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

    # load from pretrained imagenet
    if config["pretrained_imagenet2k"]:
        print(f"From pretrained at {config['pretrained_imagenet2k']}")
        model.load_from_imagenet(np.load(config['pretrained_imagenet2k']), config)
        
    # load from pretrained MAE
    if config["pretrained_mae"]:
        print(f"From pretrained at {config['pretrained_mae']}")
        model.load_from_mae(torch.load(config['pretrained_mae']))

    # resume training from checkpoint if indicated
    if config["checkpoint"]:
        print(f"Resuming training of {config_id} from {config['checkpoint']}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, config['checkpoint']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_metric = checkpoint['best_metric']
        start_epoch = checkpoint['epoch'] + 1
        last_global_step = checkpoint['global_step']
    else:
        best_metric = -1
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

    train(config, config_id, model, epoch_range, last_global_step, train_loader, val_loader, device, criterion, 
          best_metric, optimizer, scheduler, stopper, writer, model_dir, checkpoint_dir)

def run_test(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda')

    # Data paths
    test_dir = config["test_dir"]
    test_seq_dir = [os.path.join(test_dir, f"T{t}") for t in range(5)]
    img_names = [os.path.basename(i) for i in glob.glob(os.path.join(test_dir, "T0", "*.png"))]
    label_df = pd.read_csv(config["test_label"])

    # Datasets and loaders
    time_range = (0, 4)
    test_img_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = TumorCifarDataset(test_dir, img_names, label_df, time_range, img_transform=test_img_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2)

    model_class = model_classes[config["model_class"]]
    model = model_class(
        image_size=32,
        patch_size=config["patch_size"],
        num_classes=2,
        dim=config["embedding_dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        qkv_bias=config["qkv_bias"],
        time_embedding=config["time_emb"],
        pos_embedding=config["pos_emb"],
        dim_head=config["dim_head"],
    ).to(device)

    # load best model
    model_dir = os.path.join(config["root_dir"], "models", config_id)
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path))

    test(model, test_loader, device, model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    config_dir = "/home/local/VANDERBILT/litz/github/MASILab/time-distance-transformer/configs"
    config = load_config(config_dir, args.config)
    if args.train:
        run_train(config, args.config)
    if args.test:
        run_test(config, args.config)