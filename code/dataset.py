import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time
import glob
from utils import load_images
import random
import numpy as np
import cv2

class BrainTumor(Dataset):
    def __init__(self, path, split='train', validation_split=0.2, img_size=256, length=64):
        self.path = path
        train_data=pd.read_csv(os.path.join(self.path, 'train_labels.csv'))
        self.labels={}
        brats=list(train_data['BraTS21ID'])
        mgmt=list(train_data['MGMT_value'])
        self.length=length
        self.img_size=img_size
        self.trans = False
        for b,m in zip(brats,mgmt):
            self.labels[str(b).zfill(5)]=m
        if split=='valid':
            self.split='train'
            self.ids=[a.split("/")[-1] for a in sorted(glob.glob(path+f"/{self.split}/"+"*"))]
            self.ids=self.ids[:int(len(self.ids)*validation_split)]
            self.trans = False
        elif split=='train':
            self.split = split
            self.ids = [a.split("/")[-1] for a in sorted(glob.glob(path + f"/{self.split}/" + "*"))]
            self.ids = self.ids[int(len(self.ids) * validation_split):]
            self.trans = True
        else:
            self.split = split
            self.ids = [a.split("/")[-1] for a in sorted(glob.glob(path + f"/{self.split}/" + "*"))]
            self.trans = False


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        images = load_images(self.path, self.ids[idx],self.split,img_size=self.img_size,length=self.length, trans=self.trans)
        images = images - images.mean()
        images = (images+1e-5)/(images.std()+1e-5)

        if self.split=='test':
            return torch.tensor(images,dtype=torch.float32),self.ids[idx]
        else:
            labels=self.labels[self.ids[idx]]
            return torch.tensor(images,dtype=torch.float32),torch.tensor(labels,dtype=torch.long)

    def reset_seed(self, epoch, seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print("-------- TEST DATASET --------")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_dataset = BrainTumor(
        path="/data/zhaoxinying/datasets/dataset",
        split="train",
        validation_split=0.2,
        img_size=256,
        length=64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    time_start = time.time()
    print("START", time_start)
    for epoch in (1, 10):
        train_loader.dataset.reset_seed(epoch, 777)
        for i, (img, label) in enumerate(train_loader):
            time_end = time.time()
            print(time_end - time_start)
            time_start = time.time()