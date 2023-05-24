import os
import sys
import glob
import numpy as np
import math
import pandas as pd
from pathlib import Path
import argparse
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate # https://github.com/pytorch/pytorch/issues/11372

from utils import seed_everything, load_config, EarlyStopper, get_cv_folds
from datasets import NLSTDataset, NLSTDatasetFromFeat, NLSTDatasetFromROI
from ViT3D import SimpleViT3D
from TimeDistanceViT import TimeDistanceViT
from FeatViT import FeatViT, TimeAwareFeatViT
from MAE import MAE
from train import train, pretrain_MAE
from test import test
from scheduler import ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineSchedule
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    ToTensor,
    Resize,
    RandShiftIntensity,
    RandAffine,
    RandRotate,
)

model_classes = {
    "ViT3D": SimpleViT3D,
    "TimeDistanceViT": TimeDistanceViT,
    "FeatViT": FeatViT,
    "TimeAwareFeatViT": TimeAwareFeatViT,
}
schedules = {
    "Constant": ConstantLRSchedule,
    "WarmupConstant": WarmupConstantSchedule,
    "Linear": WarmupLinearSchedule,
    "Cosine": WarmupCosineSchedule,
}

def run_cv_train(config, config_id, kfold):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    train_dir = config["train_dir"]
    label_df = pd.read_csv(config["label"])
    cv_path = config["cv"]
    log_dir = os.path.join(config["root_dir"], "logs", config_id, f"fold{kfold}")
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id, f"fold{kfold}")
    model_dir = os.path.join(config["root_dir"], "models", config_id, f"fold{kfold}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # cv folds
    with open(cv_path, 'rb') as handle:
        cv = pickle.load(handle)
    train_pids, _ = get_cv_folds(cv, kfold)

    # Training/Validation split
    val_size = int(np.floor(config["val_fraction"] * len(train_pids)))
    np.random.shuffle(train_pids)
    val_pids, train_pids = train_pids[:val_size], train_pids[val_size:]

    # Datasets and loaders
    time_range = (0, 1)
    ss = config["image_size"]
    if config["dataset"] == "NLSTDatasetFromROI":
        train_img_transforms = Compose([
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            RandShiftIntensity(offsets=0.10, prob=0.2),
            RandAffine(mode='bilinear', 
                       prob=0.5, spatial_size=[ss, ss, ss], 
                       rotate_range=(0,0,np.pi/30),
                       scale_range=(0.2, 0.2, 0.2),),
            # RandRotate((0,np.pi/4,0), prob=0.2),
            ToTensor(),
        ])
        val_img_transforms = Compose([
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        train_dataset = NLSTDatasetFromROI(train_dir, train_pids, label_df, pbb_dir=config["pbb_dir"], patch_size=config["image_size"],
                                time_range=time_range,topk=config["topk"], img_transform=train_img_transforms, phase="train")
        val_dataset = NLSTDatasetFromROI(train_dir, val_pids, label_df, pbb_dir=config["pbb_dir"], patch_size=config["image_size"],
                                time_range=time_range, topk=config["topk"],img_transform=val_img_transforms, phase="train")
    
    elif config["dataset"] == "NLSTDatasetFromFeat":
        train_img_transforms = ToTensor()
        val_img_transforms = ToTensor()
        train_dataset = NLSTDatasetFromFeat(train_dir, train_pids, label_df, feat_dim=64,time_range=time_range,topk=config["topk"], 
            img_transform=train_img_transforms)
        val_dataset = NLSTDatasetFromFeat(train_dir, val_pids, label_df, feat_dim=64, time_range=time_range, topk=config["topk"], 
            img_transform=val_img_transforms)
    
    else:
        train_img_transforms = Compose([
            Resize((256, 256, 256), mode="area"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            # RandShiftIntensity(offsets=0.10, prob=0.2),
            # RandAffine(mode='bilinear', 
            #            prob=0.5, spatial_size=[ss, ss, ss], 
            #            rotate_range=(0,0,np.pi/30),
            #            scale_range=(0.2, 0.2, 0.2),),
            # RandRotate((0,np.pi/4,0), prob=0.2),
            ToTensor(),
        ])
        val_img_transforms = Compose([
            Resize((256, 256, 256), mode="area"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        train_dataset = NLSTDataset(train_dir, train_pids, label_df, time_range, img_transform=train_img_transforms)
        val_dataset = NLSTDataset(train_dir, val_pids, label_df, time_range, img_transform=val_img_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), num_workers=3, pin_memory=True)

    # model
    model_class = model_classes[config["model_class"]]
    if (config["model_class"]=="FeatViT") or (config["model_class"]=="TimeAwareFeatViT"):
        model = model_class(
            num_feat=config["topk"],
            feat_dim=config["feat_dim"],
            num_classes=2,
            dim=config["embedding_dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            qkv_bias=config["qkv_bias"],
            time_embedding=config["time_emb"],
        ).to(device)
    else:
        model = model_class(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_classes=2,
            dim=config["embedding_dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            qkv_bias=config["qkv_bias"],
            patch_embedding=config["patch_emb"],
            time_embedding=config["time_emb"],
            pos_embedding=config["pos_emb"],
            channels=config["topk"],
            dim_head=config["dim_head"],
            phase="train"
        ).to(device)

    # loss function
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

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

######################################################
# Testing
######################################################
def run_test(config, config_id, kfold):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    train_dir = config["train_dir"]
    label_df = pd.read_csv(config["label"])
    cv_path = config["cv"]
    log_dir = os.path.join(config["root_dir"], "logs", config_id, f"fold{kfold}")
    model_dir = os.path.join(config["root_dir"], "models", config_id, f"fold{kfold}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # cv folds
    with open(cv_path, 'rb') as handle:
        cv = pickle.load(handle)
    _, test_pids = get_cv_folds(cv, kfold)

    # Datasets and loaders
    time_range = (0, 1)
    ss = config["image_size"]
    if config["dataset"] == "NLSTDatasetFromROI":
        test_img_transforms = Compose([
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        test_dataset = NLSTDatasetFromROI(train_dir, test_pids, label_df, pbb_dir=config["pbb_dir"], patch_size=config["image_size"],
                                time_range=time_range,topk=config["topk"], img_transform=test_img_transforms, phase="test")
    
    elif config["dataset"] == "NLSTDatasetFromFeat":
        test_img_transforms = ToTensor()
        test_dataset = NLSTDatasetFromFeat(train_dir, test_pids, label_df, feat_dim=64,time_range=time_range,topk=config["topk"], 
            img_transform=test_img_transforms)
    else:
        test_img_transforms = Compose([
            Resize((256, 256, 256), mode="area"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        test_dataset = NLSTDataset(train_dir, test_pids, label_df, time_range, img_transform=test_img_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=3, pin_memory=True)

    model_class = model_classes[config["model_class"]]
    if (config["model_class"]=="FeatViT") or (config["model_class"]=="TimeAwareFeatViT"):
        model = model_class(
            num_feat=config["topk"],
            feat_dim=config["feat_dim"],
            num_classes=2,
            dim=config["embedding_dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            qkv_bias=config["qkv_bias"],
            time_embedding=config["time_emb"],
        ).to(device)
    else:
        model = model_class(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_classes=2,
            dim=config["embedding_dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            qkv_bias=config["qkv_bias"],
            patch_embedding=config["patch_emb"],
            time_embedding=config["time_emb"],
            pos_embedding=config["pos_emb"],
            channels=config["topk"],
            dim_head=config["dim_head"],
            phase="train"
        ).to(device)

    # load best model
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path))

    test(model, test_loader, device, model_dir)

######################################################
# MAE Pretraining 
######################################################
def run_pretrain(config, config_id, kfold):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    train_dir = config["train_dir"]
    label_df = pd.read_csv(config["label"])
    cv_path = config["cv"]
    log_dir = os.path.join(config["root_dir"], "logs", config_id, f"fold{kfold}")
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id, f"fold{kfold}")
    model_dir = os.path.join(config["root_dir"], "models", config_id, f"fold{kfold}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # cv folds
    with open(cv_path, 'rb') as handle:
        cv = pickle.load(handle)
    train_pids, test_pids = get_cv_folds(cv, kfold)

    # Training/Validation split
    val_size = int(np.floor(config["val_fraction"] * len(train_pids)))
    np.random.shuffle(train_pids)
    val_pids, train_pids = train_pids[:val_size], train_pids[val_size:]

    # Datasets and loaders
    time_range = (0, 1)
    ss = config["image_size"]
    if config["dataset"] == "NLSTDatasetFromROI":
        train_img_transforms = Compose([
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            RandShiftIntensity(offsets=0.10, prob=0.2),
            RandAffine(mode='bilinear', 
                       prob=0.5, spatial_size=[ss, ss, ss], 
                       rotate_range=(0,0,np.pi/30),
                       scale_range=(0.2, 0.2, 0.2),),
            # RandRotate((0,np.pi/4,0), prob=0.2),
            ToTensor()
        ])
        val_img_transforms = Compose([
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        train_dataset = NLSTDatasetFromROI(train_dir, train_pids, label_df, pbb_dir=config["pbb_dir"], 
            patch_size=config["image_size"], time_range=time_range, topk=config["topk"], img_transform=train_img_transforms, phase="train")
        val_dataset = NLSTDatasetFromROI(train_dir, val_pids, label_df, pbb_dir=config["pbb_dir"], 
            patch_size=config["image_size"], time_range=time_range, topk=config["topk"], img_transform=val_img_transforms, phase="train")
    else:
        train_img_transforms = Compose([
            Resize((256, 256, 256), mode="area"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            RandShiftIntensity(offsets=0.10, prob=0.2),
            RandAffine(mode='bilinear', 
                       prob=0.5, spatial_size=[ss, ss, ss], 
                       rotate_range=(0,0,np.pi/30),
                       scale_range=(0.2, 0.2, 0.2),),
            # RandRotate((0,np.pi/4,0), prob=0.2),
            ToTensor(),
        ])
        val_img_transforms = Compose([
            Resize((256, 256, 256), mode="area"),
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor(),
        ])
        train_dataset = NLSTDataset(train_dir, train_pids, label_df, time_range, img_transform=train_img_transforms)
        val_dataset = NLSTDataset(train_dir, val_pids, label_df, time_range, img_transform=val_img_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), num_workers=3)

    # model
    model_class = model_classes[config["model_class"]]
    vit = model_class(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        num_classes=1000,
        dim=config["embedding_dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        qkv_bias=config["qkv_bias"],
        patch_embedding=config["patch_emb"],
        time_embedding=config["time_emb"],
        pos_embedding=config["pos_emb"],
        channels=config["topk"],
        dim_head=config["dim_head"],
        phase="pretrain"
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
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--kfold', type=str)
    args = parser.parse_args()

    config_dir = "/home/local/VANDERBILT/litz/github/MASILab/time-distance-transformer/configs"
    config = load_config(config_dir, args.config)
    # torch.multiprocessing.set_start_method('spawn') # https://github.com/pytorch/pytorch/issues/40403
    if args.pretrain:
        run_pretrain(config, args.config, args.kfold)
    if args.train:
        run_cv_train(config, args.config, args.kfold)
    if args.test:
        run_test(config, args.config, args.kfold)