import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.utils as vutils

import config
from utils import ext_transformer as et
from utils import scheduler as scheduler
from utils import metricsUtil as metricsUtil
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

    # 这里使用backbone+deeplabv3+
    if config.backbone == 'resnet50':
        model = load_model('resnet50', num_classes=9, output_stride=config.output_stride)
    else:
        model = load_model('mobilenetv2', num_classes=9, output_stride=config.output_stride)

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

    # 设置验证集评估指标集
    metric = metricsUtil.eval_metrics(config.num_classes)

    # 加载预训练模型
    curr_epoch = 0
    best_score = 0.0    # 以验证集的mean_IOU算
    if len(config.pretrained_model) > 0 and os.path.isfile(config.pretrained_model):
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)

        model.load_state_dict(checkpoint["model_state"])
        if config.use_gpu and config.num_gpu > 1:  # 允许使用GPU，才能使用多卡训练
            model = nn.DataParallel(model)
        model.to(config.device)
        if config.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])    # num_classes改变后不能接着上次训练
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            curr_epoch = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state loaded from %s" % config.pretrained_model)
        print("Load pretrained model from %s" % config.pretrained_model)
        del checkpoint
    else:
        print("[!] Retrain")
        if config.use_gpu and config.num_gpu > 1:    # 允许使用GPU，才能使用多卡训练
            model = nn.DataParallel(model)
        model.to(config.device)

    model.classifier.classifier[3] = nn.Conv2d(256, config.num_classes, kernel_size=1, stride=1)  # 加载上预训练模型后再修改最后一层

    if os.path.exists("./checkpoint/") is False:
        os.makedirs("./checkpoint/")

    train_loss = 99999  # 记录当前train loss
    eval_loss = 99999  # 记录当前eval loss

    if config.use_gpu and torch.cuda.is_available():
        model.cuda()    # 手动将model放到cuda上

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
            print("Epoch: %d, Batch: %d, train_loss: %.6f" % (epoch, i, loss))
            loss.backward()
            optimizer.step()

        curr_train_loss = loss.detach().cpu().numpy()
        # 按训练集loss的更新保存
        if curr_train_loss < train_loss:
            train_loss = curr_train_loss  # 更新保存的loss

        eval_losses = []    # 统计验证集的损失
        model.eval()
        metric.reset()    # 每轮训练完验证时都要重置混淆矩阵
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                images, labels = batch[:]
                h, w = images.shape[2], images.shape[3]
                # print(i, h, w)
                images = images.to(config.device, dtype=torch.float32)  # BCHW

                # eval时按图像本身大小算
                label_copy = labels.clone()
                labels = labels.view(-1, h * w)  # labels改变维度，变为[batch_size, 513 * 513]
                labels = labels.to(config.device, dtype=torch.long)  # [batch_size, 513 * 513]

                preds = model(images)
                loss = criterion(preds.view(-1, config.num_classes, h * w), labels)
                eval_losses.append(loss.detach().cpu().numpy())

                # 验证集采用mean_IOU衡量，直接用交叉熵损失不能真实的反映出语义分割的效果
                metric.update(labels.cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())
                val_score = metric.get_results()    # 获取统计的指标

                if i < config.val_preview_num:    # 保存前val_preview_num张图像供预览
                    outputs = preds.max(1)[1][0]

                    # 调色板
                    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                    colors = torch.as_tensor([i for i in range(config.num_classes)])[:, None] * palette
                    colors = (colors % 255).numpy().astype("uint8")

                    # 给推理结果上色
                    r_output = Image.fromarray(outputs.byte().cpu().numpy()).resize((w, h))    # resize时应该写wh
                    r_output.putpalette(colors)
                    r_output = r_output.convert('RGB')
                    # print(type(r_output), np.array(r_output).shape)
                    # r.show()

                    # 给label上色
                    r_label = Image.fromarray(label_copy[0].byte().cpu().numpy()).resize((w, h))
                    r_label.putpalette(colors)
                    r_label = r_label.convert('RGB')
                    # print(type(r_label), np.array(r_label).shape)

                    # 拼接：原图，标签，推理结果
                    show_img_list = [images.detach().cpu()[0] * 255., T.PILToTensor()(r_label), T.PILToTensor()(r_output)]

                    label_show = vutils.make_grid(show_img_list, nrow=1, padding=2, normalize=True).cpu()

                    if os.path.exists(config.val_results_path) is False:
                        os.makedirs(config.val_results_path)

                    vutils.save_image(label_show, os.path.join(config.val_results_path, "eval_%d_%d.png" % (epoch, i)))

        # 保存模型
        curr_eval_loss = np.mean(eval_losses)
        print("Epoch: %d, Batch: %d, train_loss: %.6f, eval_loss: %.6f" % (epoch, len(train_dataloader), curr_train_loss, curr_eval_loss))
        # 输出验证集评估结果
        print("\tDetails:", val_score)

        if val_score['mean_IOU'] > best_score:    # 只有当前模型比之前best_score好时才保存
            best_score = val_score['mean_IOU']

            torch.save({
                "cur_itrs": epoch,
                "model_state": model.module.state_dict() if config.use_gpu and config.num_gpu > 1 else model.state_dict(),    # 多GPU才有model.nodule结构
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score
            }, './checkpoint/faceseg_%d_%.6f_%.6f_%.6f.pth' % (epoch, curr_train_loss, curr_eval_loss, best_score))
        scheduler.step()