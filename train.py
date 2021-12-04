import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils import ext_transformer as et
from utils import scheduler as scheduler
from dataset import split_dataset, HelenFace
from model import load_model

if __name__ == '__main__':
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    # 拆分训练集和验证集
    trainlist, vallist = split_dataset(config.img_path)

    train_dataset = HelenFace(trainlist, train_transform)
    val_dataset = HelenFace(vallist, val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # 这里使用resnet50+deeplabv3+
    model = load_model('resnet50')

    # backbone bn层设置动量
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
    # 设置优化器
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * config.learning_rate},
        {'params': model.classifier.parameters(), 'lr': config.learning_rate}
    ], lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

    # 设置scheduler
    scheduler = scheduler.PolyLR(optimizer, config.total_epochs, power=0.9)

    # 设置损失函数：本质上是分类问题，用交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


