# face_segmentation

采用deeplabv3plus实现人脸语义分割

## 0、deeplabv3plus网络结构

<div align='center'>
  <img src='./assets/nn_structure.png'>
</div>

## 1、模型

​	backbone支持resnet50和mobilenetv2两种，配置方式如下：

``` python
# config.py

# 设置backbone
backbone = "mobilenetv2"    # backbone，可选resnet50 或 mobilenetv2
# pretrained_model = ""    # 预训练模型设为空，表示从头开始训练
# pretrained_model = "./checkpoint/best_deeplabv3plus_resnet50_voc_os16.pth"    # backbone为resnet50的预训练模型
pretrained_model = "./checkpoint/best_deeplabv3plus_mobilenet_voc_os16.pth"    # backbone为mobilenetv2的预训练模型

# backbone的预训练模型
# model_urls = "https://download.pytorch.org/models/resnet50-19c8e357.pth"    # resnet
model_urls = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"    # mobilenetv2
```

## 2、数据集

​	采用helen-face数据集训练，人脸共分类11种类别。

``` python
face_seg_label = {
    0: "background", 
    1: "face",
    2: "right eyebrow",
    3: "left eyebrow",
    4: "right eye",
    5: "left eye",
    6: "nose",
    7: "upper lip",
    8: "tooth",
    9: "down lip", 
    10: 'hair'
}
```

​	实际上不考虑牙齿和头发，所以共分为9类，新的对应关系如下：

``` python
face_seg_label = {
    0: "background", 
    1: "face",
    2: "right eyebrow",
    3: "left eyebrow",
    4: "right eye",
    5: "left eye",
    6: "nose",
    7: "upper lip",
    8: "down lip"
}
```

