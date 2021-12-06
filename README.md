# face_segmentation

采用deeplabv3+实现人脸语义分割

## 1、模型

​	采用resnet50作为backbone，deeplabv3plus实现。

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

​	实际上不考虑牙齿和头发，所以共分类9类别，新的对应关系如下：

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

