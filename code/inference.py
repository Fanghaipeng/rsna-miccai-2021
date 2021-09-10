import os
import numpy as np
import sys
import yaml
import random
import argparse
import glob
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from models.efficientnet_gru import efficientNet_gru
from models.efficientnet_pytorch_3d.model import EfficientNet3D

from dataset import BrainTumor

torch.backends.cudnn.benchmark = True

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
    print("--- TEST ARGS ---")
    print(opt)

    if not os.path.exists("configs/%s.yaml" % opt.config):
        print("*** configs/%s.yaml not found. ***" % opt.config)
        exit()

    # read yaml config
    f = open("configs/%s.yaml" % opt.config, "r", encoding="utf-8")
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("--- CONFIG ---")
    print(config)

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

    if opt.gpu_num > 1 and opt.gpu_id ==-1:
        model = nn.DataParallel(model)
    elif opt.gpu_num > 1 and opt.gpu_id != -1:
        print("gpu_num and gpu_id not correct")
        sys.exit()

    # ---------------- PATH ----------------
    predpath = os.path.join(config["path"]["pred_path"], config["modelname"])
    os.makedirs(predpath, exist_ok=True)

    test_dir = config["test"]["test_path"]
    testlist = glob.glob(test_dir + "/*")

    csvpath = "/data/zhaoxinying/code/rsna-miccai-2021/user_data/preds/submission.csv"

    # ---------------- DATASET ----------------
    test_dataset = BrainTumor(
        path=config["path"]["data_path"],
        split="test",
        img_size=config["test"]["imgsize"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test"]["batchsize"],
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    # ---------------- LOAD MODEL ----------------
    model.load_state_dict(torch.load(config["path"]["resume_path"]))
    # model = torch.load(config["path"]["resume_path"].replace("model_best", "model_best_full"))
    model.cuda()

    model.eval()
    submission = pd.read_csv(csvpath, index_col="BraTS21ID")

    idxs = []
    preds = []

    with torch.no_grad():
        for i, (img, idx) in enumerate(tqdm(test_loader)):
            img = img.cuda()
            output = model(img)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = pred[1]
            # print(idx, pred)
            idx = int(idx[0])

            idxs.append(idx)
            preds.append(pred)


    preddf = pd.DataFrame({"BraTS21ID": idxs, "MGMT_value": preds})
    preddf = preddf.set_index("BraTS21ID")

    submission["MGMT_value"] = 0
    submission["MGMT_value"] += preddf["MGMT_value"]

    submission["MGMT_value"].to_csv(predpath + "/submission.csv")
    print("SAVE INFERENCE")