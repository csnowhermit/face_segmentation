import os
import torch

'''
    模型的配置项
'''

# 数据集相关
num_classes = 10
img_path = "E:/dataset/helen_face/data/"
anno_path = "E:/dataset/helen_face/labels/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random_seed = 1
batch_size = 16
val_batch_size = 4
learning_rate = 0.01
weight_decay = 0.0001    # 权重衰减
total_epochs = 30000    # 训练总的epochs数

# resnet50 backbone 相关
pretrained = True
progress = True
model_urls = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

