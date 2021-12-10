import os
import cv2
import time
import random
import numpy as np

# base_path = "F:/dataset/helen_face/images/"
# label_path = "F:/dataset/SmithCVPR2013_dataset_original/labels/"
#
# tag_dir = "F:/dataset/helen_face/eyebrow/"
#
# for file in os.listdir(base_path):
#     if random.randint(0, 10) % 2 == 0:
#         tag = "left"
#         label_num = "03"
#     else:
#         tag = "right"
#         label_num = "02"
#
#     tag_path = os.path.join(tag_dir, tag)
#     label_file = os.path.join(label_path, "%s/%s_lbl%s.png" % (file[0:-4], file[0:-4], label_num))
#     print(label_file)
#     start = time.time()
#     label_img = cv2.imread(label_file)    # 标签图
#     img = cv2.imread(os.path.join(base_path, file))    # 原图
#
#     img = np.where(label_img>0, img, 0)
#
#     cv2.imwrite(os.path.join(tag_dir, "%s/%s" % (tag, file)), img)
#     print("finished: ", time.time() - start)


# img = cv2.imread("F:/dataset/SmithCVPR2013_dataset_original/labels/100591971_1/100591971_1_lbl01.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = np.where(img==255, 1, 0)
# print(img.shape)
# np.savetxt("./img1.txt", np.round(img), fmt='%.1f')

# 计算语义分割任务的评价指标
hist = np.arange(1, 10).reshape(3, 3)
print(hist)

# 1.总体acc
acc = np.diag(hist).sum() / hist.sum()
print("1.总体acc", acc)

# 2.分类acc
acc_cls = np.diag(hist) / hist.sum(axis=1)
print(np.diag(hist))
print(hist.sum(axis=1))    # 横向每行累加
print("2.各类别acc", acc_cls)
acc_cls = np.nanmean(acc_cls)
print("3.各类别平均acc", acc_cls)

# 3.iou
print("hist.sum(axis=1):", hist.sum(axis=1))
print("hist.sum(axis=0):", hist.sum(axis=0))
print(np.diag(hist))
iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
print("3.iou", iou)
mean_iou = np.nanmean(iou)
print("4.各类平均iou", mean_iou)

# 4.freq
print(hist.sum(axis=1))
print(hist.sum())
freq = hist.sum(axis=1) / hist.sum()    #
print("5.标签中每种类别的像素点占比", freq)
print(freq[freq>0])    # 找出已识别到的类别
print(iou[freq>0])    # 找出已识别到的类别中每类的iou
print(freq[freq>0] * iou[freq>0])

# fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
# print(fwavacc)

# 5.各类别分别的iou
cls_iou = dict(zip(range(3), iou))
print("6.各类别的分别iou", cls_iou)
