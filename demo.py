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