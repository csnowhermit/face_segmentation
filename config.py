import torch

'''
    模型的配置项
'''

# 数据集相关
num_classes = 9
img_path = "F:/dataset/helen_face/images/"
anno_path = "F:/dataset/helen_face/labels/"
resize = 513

use_gpu = False    # 设置是否使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
num_gpu = 1    # GPU个数
random_seed = 1
batch_size = 8
val_batch_size = 1    # val时，需用label逐像素比对，直接resize后像素层面会造成误差（eval的结果需要预览，也不能按训练集那样裁剪固定大小）
val_preview_num = 10    # val时保存多少张图像供预览
learning_rate = 0.01
weight_decay = 0.0001    # 权重衰减
total_epochs = 30000    # 训练总的epochs数
continue_training = True    # 是否接着上次的训练
pre_frev = 10    # 每隔10个batch打印一次训练信息
output_stride = 16
val_results_path = "./results/"    # 验证集结果保存目录

pretrained = True
progress = True

# 设置backbone
backbone = "mobilenetv2"    # backbone，可选 resnet50 或 mobilenetv2
# pretrained_model = ""    # 预训练模型设为空，表示从头开始训练
# pretrained_model = "./checkpoint/best_deeplabv3plus_resnet50_voc_os16.pth"    # backbone为resnet50的预训练模型
pretrained_model = "./checkpoint/latest_deeplabv3plus_mobilenet_voc_os16_0.176498.pth"    # backbone为mobilenetv2的预训练模型

# backbone的预训练模型
# model_urls = "https://download.pytorch.org/models/resnet50-19c8e357.pth"    # resnet
model_urls = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"    # mobilenetv2

# 推理过程
inference_imgpath = "F:/dataset/helen_face/inference"