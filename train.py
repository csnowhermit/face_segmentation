import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from utils import ext_transformer as et
from utils import scheduler as scheduler
from utils import stream_metrics as stream_metrics
from dataset import split_dataset, HelenFace
from model import load_model

'''
    验证并输出样本
    :param val_results_path 验证结果保存路径
    :param ret_samples_ids 指定第几个batch的结果需要返回
'''
def validate(model, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []    # 返回要可视化的图像集合

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()    # 推理结果
            labels = labels.cpu().numpy()    # 标签

            metrics.update(labels, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append((images[0].detach().cpu().numpy(), labels[0], preds[0]))    # 原图，标签图，预测图
        score = metrics.get_results()
    return score, ret_samples


'''
    图像的反标准化
'''
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return F.normalize(tensor, _mean, _std)

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

    # 设置指标报表
    metrics = stream_metrics.StreamSegMetrics(config.num_classes)

    # 加载预训练模型
    curr_epoch = 0
    best_score = 0.0    # 以验证集的score算
    if len(config.pretrained_model) > 0 and os.path.isfile(config.pretrained_model):
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(config.device)
        if config.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            curr_epoch = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % config.pretrained_model)
        print("Model restored from %s" % config.pretrained_model)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(config.device)

    # 开始训练
    interval_loss = 0.0    # 用于计算从开始训练到当前epochs的平均损失用
    for epoch in range(curr_epoch, config.total_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):    # 批量加载数据训练
            images, labels = batch[:]
            images = images.to(config.device, dtype=torch.float32)
            labels = labels.to(config.device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            # 每隔多少个batch打印下损失
            if i % config.pre_frev == 0:
                interval_loss = interval_loss / config.pre_frev
                print("Epoch: %d, batch: %d, Loss: %.6f, avg_loss: %.6f" % (epoch, i, np_loss, interval_loss))
                interval_loss = 0.0    # 打印之后置为0，重新计数损失

            if i % config.val_interval == 0:
                print("Validation...")
                torch.save({
                    "cur_itrs": i,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
                }, './checkpoint/best_deeplabv3plus_resnet50_os%d.pth' % (config.output_stride))

                model.eval()

            # model, loader, device, metrics, val_results_path, ret_samples_ids = None
            vis_sample_id = np.random.randint(0, len(val_dataloader), config.vis_num_samples, np.int32) if config.enable_vis else None    # 设置允许第几个batch可视化
            val_score, ret_samples = validate(model=model, loader=val_dataloader, device=config.device, metrics=metrics, val_results_path=config.val_results_path, ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))

            # 保存最优模型
            if val_score['Mean_IOU'] > best_score:
                best_score = val_score['Mean_IOU']
                torch.save({
                    "cur_itrs": i,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
                }, './checkpoint/latest_deeplabv3plus_resnet50_os%d.pth' % (config.output_stride))

            if len(config.val_results_path) > 0:
                if os.path.exists(config.val_results_path) is False:
                    os.makedirs(config.val_results_path)

                for k, (img, label, preds) in enumerate(ret_samples):
                    img = (denormalize(img) * 255).astype(np.uint8)
                    label = (denormalize(label) * 255).astype(np.uint8)
                    pred = (denormalize(preds) * 255).astype(np.uint8)

                    concat_img = np.concatenate((img, label, pred), axis=2)    # 拼接


            model.train()
        scheduler.step()
