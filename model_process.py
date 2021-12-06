import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from utils import ext_transformer as et
from utils import scheduler as scheduler
from dataset import split_dataset, HelenFace
from model import load_model

if __name__ == '__main__':
    # 这里使用resnet50+deeplabv3+
    model = load_model('resnet50', num_classes=21, output_stride=config.output_stride)
    # print(model)

    checkpoint = torch.load(config.pretrained_model, map_location=config.device)
    model.load_state_dict(checkpoint["model_state"])

    model.classifier.classifier[3] = nn.Conv2d(256, 10, kernel_size=1, stride=1)

    print(model)