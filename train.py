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
    train_transform = et.ExtCompose([
        # et.ExtResize(size=config.resize),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(config.resize, config.resize), pad_if_needed=True),
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

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True)

    # 这里使用resnet50+deeplabv3+
    model = load_model('resnet50', num_classes=config.num_classes, output_stride=config.output_stride)

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

    # 加载预训练模型
    curr_epoch = 0
    # best_score = 0.0    # 以验证集的score算
    if len(config.pretrained_model) > 0 and os.path.isfile(config.pretrained_model):
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(config.device)
        if config.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            curr_epoch = checkpoint["cur_itrs"]
            # best_score = checkpoint['best_score']
            print("Training state restored from %s" % config.pretrained_model)
        print("Model restored from %s" % config.pretrained_model)
        del checkpoint
    else:
        print("[!] Retrain")
        if config.use_gpu and config.num_gpu > 1:    # 允许使用GPU，才能使用多卡训练
            model = nn.DataParallel(model)
        model.to(config.device)

    if os.path.exists("./checkpoint/") is False:
        os.makedirs("./checkpoint/")

    train_loss = 99999  # 记录当前train loss
    eval_loss = 99999  # 记录当前eval loss

    # 开始训练
    for epoch in range(curr_epoch, config.total_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):    # 批量加载数据训练
            images, labels = batch[:]
            images = images.to(config.device, dtype=torch.float32)    # [16, 3, 513, 513]

            labels = labels.view(-1, config.resize * config.resize)    # labels改变维度，变为[batch_size, 513 * 513]
            labels = labels.to(config.device, dtype=torch.long)    # [batch_size, 513 * 513]

            optimizer.zero_grad()
            output = model(images)    # [16, num_classes, 513, 513]
            loss = criterion(output.view(-1, config.num_classes, config.resize * config.resize), labels)
            loss.backward()
            optimizer.step()
            break

        curr_train_loss = loss.detach().cpu().numpy()
        # 按训练集loss的更新保存
        if curr_train_loss < train_loss:
            train_loss = curr_train_loss  # 更新保存的loss

        eval_losses = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):  # 批量加载数据训练
                images, labels = batch[:]
                h, w = images.shape[2], images.shape[3]
                # print(i, h, w)
                images = images.to(config.device, dtype=torch.float32)  # [16, 3, 513, 513]

                # eval时按图像本身大小算
                labels = labels.view(-1, h * w)  # labels改变维度，变为[batch_size, 513 * 513]
                labels = labels.to(config.device, dtype=torch.long)  # [batch_size, 513 * 513]

                preds = model(images)
                loss = criterion(preds.view(-1, config.num_classes, h * w), labels)
                eval_losses.append(loss.detach().cpu().numpy())

                if i < config.val_preview_num:    # 保存前val_preview_num张图像供预览
                    print("这里保存下eval的效果")

        # 保存模型
        curr_eval_loss = np.mean(eval_losses)
        if curr_eval_loss < eval_loss:
            eval_loss = curr_eval_loss

            torch.save({
                "cur_itrs": epoch,
                "model_state": model.module.state_dict() if config.use_gpu else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": 0.0
            }, './checkpoint/faceseg_%d_%.6f_%.6f.pth' % (epoch, curr_train_loss, curr_eval_loss))
        scheduler.step()
