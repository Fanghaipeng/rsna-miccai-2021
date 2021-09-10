import os
import torch

from models.efficientnet_gru import efficientNet_gru
from models.efficientnet_pytorch_3d.model import EfficientNet3D

from dataset import BrainTumor


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = EfficientNet3D.from_name(
        "efficientnet-b0",
        override_params={'num_classes': 2},
        in_channels=4)
    criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load("../user_data/models/efficientnet3d_b0_lr0.0003_aug256_2/model_best.pt"))
    torch.save(model, "../user_data/models/efficientnet3d_b0_lr0.0003_aug256_2/model_best_full.pt")
    print("SAVE!")

    model.cuda()
    model.eval()
    inputs = torch.randn((1, 1, 200, 200, 200)).cuda()
    labels = torch.tensor([1]).cuda()
    outputs = model(inputs)
    tmp = outputs.squeeze(1)
    print("tmp", tmp)
    print("outputs", outputs)



    loss = criterion(outputs, labels)
    print("loss", loss)
