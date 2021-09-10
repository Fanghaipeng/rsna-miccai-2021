import os
import numpy as np
import sys
import yaml
import json
import random
import argparse
import shutil
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from models.efficientnet_gru import efficientNet_gru
from models.efficientnet_pytorch_3d.model import EfficientNet3D

from dataset import BrainTumor
from utils import one_hot
from utils import Progbar
from schedulers import create_optimizer, default_scheduler

torch.backends.cudnn.benchmark = True


def str2bool(in_str):  # transform strings in config to bool
    if in_str in [1,"1", "t", "True", "true"]:
        return True
    elif in_str in [0,"0", "f", "False", "false", "none"]:
        return False


if __name__ == "__main__":
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model1")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--manualSeed', type=int, default=-1)
    parser.add_argument("--gpu_id", default=-1)
    parser.add_argument('--gpu_num', type=int, default=1)
    opt = parser.parse_args()

    # print args in train.sh
    print("--- TRAINING ARGS ---")
    print(opt)

    if not os.path.exists("configs/%s.yaml" % opt.config):
        print("*** configs/%s.yaml not found. ***" % opt.config)
        exit()

    # read yaml config
    f = open("configs/%s.yaml" % opt.config, "r", encoding="utf-8")
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("--- CONFIG ---")
    print(config)

    # random seed
    if opt.manualSeed == -1:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.determinstic = True
    cudnn.benchmark = True

    # paths
    model_savedir = os.path.join(config["path"]["model_dir"], config["modelname"])
    os.makedirs(model_savedir, exist_ok=True)
    params = vars(opt)
    params_file = os.path.join(model_savedir, "params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp, indent=4)

    # tensorboard
    writer_dir = os.path.join(config["path"]["writer_path"], config["modelname"])
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir, ignore_errors=True)
    writer = SummaryWriter(logdir=writer_dir)

    # ---------------- CHOOSE MODEL ----------------
    if "efficientnet3d" in config["describe"]:
        model = EfficientNet3D.from_name(
            "efficientnet-b0",
            override_params={'num_classes': 2},
            in_channels=4)
        print("--- Use model Efficientnet 3d ---")

    if "efficientnet_gru" in config["describe"]:
        model = efficientNet_gru(
            in_channels=4,
            out_channels=2,
            hidden_channels=100,
            image_size=256,
            length=config["train"]["length"])
        print("--- Use model Efficientnet-GRU ---")

    # GPU settings
    if opt.gpu_id != -1:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    elif opt.gpu_num > 1:
        pass
    model.cuda()
    if opt.gpu_num > 1 and opt.gpu_id ==-1:
        model = nn.DataParallel(model)
    elif opt.gpu_num > 1 and opt.gpu_id != -1:
        print("gpu_num and gpu_id not correct")
        sys.exit()

    # ---------------- DATASET ----------------
    train_dataset = BrainTumor(
        path=config["path"]["data_path"],
        split='train',
        validation_split=0.2,
        img_size=config["train"]["imagesize"],
        length=config["train"]["length"])
    valid_dataset = BrainTumor(
        path=config["path"]["data_path"],
        split='valid',
        validation_split=0.2,
        img_size=config["train"]["imagesize"],
        length=config["train"]["length"])
    print("length of train dataset", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batchsize"]),
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["test"]["batchsize"]),
        shuffle=False,
        num_workers=opt.workers)
    print("Trainset Size: {}; Validset Size: {}".format(len(train_dataset), len(valid_dataset)))

    # ---------------- LOSS AND OPTIM ----------------
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.08)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.25)
    default_scheduler["batch_size"] = int(config["train"]["batchsize"])
    default_scheduler["learning_rate"] = float(config["train"]["lr"])
    default_scheduler["schedule"]['params']['max_iter'] = len(train_loader)
    optimizer, scheduler = create_optimizer(optimizer_config=default_scheduler, model=model)

    # ---------------- LOAD EXISTED MODEL ----------------
    # 暂时先不写resume这个了，估计可能也可以不太用它？

    best_roc = 0.0
    k = 0
    for epoch in range(0, int(config["train"]["niter"])):
        train_loader.dataset.reset_seed(epoch, 777)

        # ---------------- TRAIN ----------------
        model.train()

        progbar_train = Progbar(
            len(train_loader), stateful_metrics=["epoch", "config", "lr"]
        )
        running_loss = 0.0
        label_all = []
        output_all = []
        roc = 0.0
        train_loss = []
        for img, label in train_loader:
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss.append(loss.item())
            output_all.extend(output.tolist())
            label_all.extend(label.tolist())

            # tensorboard
            if k % 5 == 0 and k > 0:
                writer.add_scalars(config["modelname"], {"loss": loss.item(), "roc":roc_auc_score(one_hot(label_all), output_all)}, k)
            k = k + 1

            # progbar
            progbar_train.add(1, values=[("epoch", epoch),
                                         ("loss", loss.item()),
                                         ("roc", roc_auc_score(one_hot(label_all), output_all))])

        roc += roc_auc_score(one_hot(label_all), output_all)
        print('EPOCH<', epoch, '>: train loss:', running_loss, ' train roc:', roc)

        # ---------------- VALIDATION ----------------
        model.eval()
        progbar_val = Progbar(
            len(valid_loader), stateful_metrics=["epoch", "config", "lr"]
        )

        valid_loss = 0.0
        valid_label_all = []
        valid_output_all = []
        valid_roc = 0.0
        for img, label in valid_loader:
            img = img.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)

            valid_loss += loss.item()
            valid_output_all.extend(output.tolist())
            valid_label_all.extend(label.tolist())

            # # tensorboard
            # if k % 5 == 0 and k > 0:
            #     writer.add_scalars(config["modelname"],
            #                        {"val_loss": loss.item()}, k)

            # progbar
            progbar_val.add(1, values=[("epoch", epoch),
                                         ("loss", loss.item())])
        writer.add_scalars("loss", {"train_loss": running_loss, "val_loss": valid_loss}, epoch)

        valid_roc = roc_auc_score(one_hot(valid_label_all), valid_output_all)
        scheduler.step()
        print('EPOCH<', epoch, '>: val loss:', valid_loss, ' val roc:', valid_roc)

        # ---------------- SAVE MODEL ----------------
        # save model at every step
        torch.save(model.state_dict(), os.path.join(model_savedir, "model_epoch%d_%.4f.pt" % (epoch, valid_roc)))

        # best model
        if valid_roc > best_roc:
            best_roc = valid_roc
            torch.save(model.state_dict(), os.path.join(model_savedir, "model_best.pt"))