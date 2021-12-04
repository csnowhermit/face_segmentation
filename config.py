import os
import torch

'''
    模型的配置项
'''

# 数据集相关
num_classes = 1000
img_path = "E:/dataset/helen_face/data/"
anno_path = "E:/dataset/helen_face/labels/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random_seed = 1
batch_size = 16
val_batch_size = 4

# resnet50 backbone 相关
pretrained = False
progress = False
model_urls = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

