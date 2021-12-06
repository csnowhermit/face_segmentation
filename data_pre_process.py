import os
import cv2
import numpy as np

'''
    helen-face数据集标签制作：根据分割图做成8位图
    牙齿8和头发10不要
'''

# face_seg_label = {
#     0: "background",
#     1: "face",
#     2: "right eyebrow",
#     3: "left eyebrow",
#     4: "right eye",
#     5: "left eye",
#     6: "nose",
#     7: "upper lip",
#     8: "tooth",
#     9: "down lip",
#     10: 'hair'
# }

base_path = "E:/dataset/SmithCVPR2013_dataset_original/labels/"

for annodir in os.listdir(base_path):
    annopath = os.path.join(base_path, annodir)
    img = cv2.imread(os.path.join(annopath, "%s_lbl00.png" % annodir))    # 232194_1_lbl00.png
    h, w = img.shape[0], img.shape[1]
    del img
    res = []
    filelist = os.listdir(annopath)
    num = 0
    for i in range(len(filelist)):
        # filename = 1629243_1_lbl03.png
        if i == 8 or i == 10:
            continue
        filename = "%s_lbl%02d.png" % (annodir, i)
        print(filename)
        img = cv2.imread(os.path.join(annopath, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.where(img>0, num, 0)
        num += 1
        res.append(img)

    # for file in os.listdir(annopath):
    #     if file.endswith("00.png"):
    #         continue
    #     print(os.path.join(annopath, file), file)
    #     img = cv2.imread(os.path.join(annopath, file))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     num = int(file.split(".")[0].split("_")[2][3:])
    #     img = np.where(img > 0, num, 0)
    #     res.append(img)
    #
    res = np.array(res)
    res = res.max(0)
    print(type(res), res.shape)

    res = res.reshape(h, w, 1)
    cv2.imwrite(os.path.join("E:/dataset/helen_face/labels/", "%s.png" % annodir), res)

