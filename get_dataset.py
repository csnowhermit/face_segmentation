import torch
from torch.utils.data import DataLoader

import config
from utils import ext_transformer as et
from dataset import split_dataset, HelenFace

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

    for i, batch in enumerate(train_dataloader):  # 批量加载数据训练
        images, labels = batch[:]
        print(images.shape, labels.shape, end=' ')    # torch.Size([16, 3, 513, 513]) torch.Size([16, 513, 513])

        # 模型输出是[16, num_classes, 513, 513]，要跟labels算损失，则labels得增加维度
        labels = torch.unsqueeze(labels, 1)    # labels增加维度，变为[16, 1, 513, 513]

        print(labels.shape)

    # import cv2
    # from PIL import Image
    # import numpy as np

    # label = Image.open("E:/dataset/helen_face/labels/10405146_1.png")
    # img = np.array(label)
    # print(img.shape)