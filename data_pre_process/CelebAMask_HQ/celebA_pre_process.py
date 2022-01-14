import os
import cv2
import numpy as np

# 眉毛
#     00000_l_brow.png
#     00000_r_brow.png
# 眼睛
#     00000_l_eye.png
#     00000_r_eye.png
# 鼻子
#     00000_nose.png
# 上嘴唇
#     00000_u_lip.png
# 下嘴唇
#     00000_l_lip.png
# 脸
#     00000_skin.png
# 背景
#     0

# # 7类
# {
#     0: 'background',
#     1: 'face',
#     2: 'brow',
#     3: 'eye',
#     4: 'nose',
#     5: 'up_lip',
#     7: 'down_lip'
# }

# # 8类
# {
#     0: 'background',
#     1: 'face',
#     2: 'brow',
#     3: 'eye',
#     4: 'nose',
#     5: 'up_lip',
#     6: 'mouth',
#     7: 'down_lip'
# }

# # 10类
# {
#     0: 'background',
#     1: 'face',
#     2: 'left_brow',
#     3: 'right_brow',
#     4: 'left_eye',
#     5: 'right_eye',
#     6: 'nose',
#     7: 'up_lip',
#     8: 'mouth',
#     9: 'down_lip'
# }

if __name__ == '__main__':
    base_path = "F:/dataset/CelebAMask-HQ/images/"
    label_path = "F:/dataset/CelebAMask-HQ/origin_mask/"    # 这里需要将原始标注目录CelebAMask-HQ-mask-anno/的所有*.png文件复制到label_path目录下

    cnt = 0
    for file in os.listdir(base_path):
        cnt += 1
        if cnt % 100 == 0:
            print("Processing... ", cnt)
        res = []


        # 1.先拼接脸
        if os.path.exists(os.path.join(label_path, "%05d_skin.png" % int(file[0:-4]))):
            img_face = cv2.imread(os.path.join(label_path, "%05d_skin.png" % int(file[0:-4])))
            h, w = img_face.shape[0], img_face.shape[1]

            img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)    # 没脸才需要把这注释掉。有脸时背景是0，脸为1，之后递增
            img_face = np.where(img_face > 0, 1, 0)
            res.append(img_face)

        # 2.眉毛
        if os.path.exists(os.path.join(label_path, "%05d_l_brow.png" % int(file[0:-4]))):
            img_brow_left = cv2.imread(os.path.join(label_path, "%05d_l_brow.png" % int(file[0:-4])))
            img_brow_left = cv2.cvtColor(img_brow_left, cv2.COLOR_BGR2GRAY)
            img_brow_left = np.where(img_brow_left > 0, 2, 0)
            res.append(img_brow_left)

        if os.path.exists(os.path.join(label_path, "%05d_r_brow.png" % int(file[0:-4]))):
            img_brow_right = cv2.imread(os.path.join(label_path, "%05d_r_brow.png" % int(file[0:-4])))
            img_brow_right = cv2.cvtColor(img_brow_right, cv2.COLOR_BGR2GRAY)
            img_brow_right = np.where(img_brow_right > 0, 2, 0)
            res.append(img_brow_right)

        # 3.眼睛
        if os.path.exists(os.path.join(label_path, "%05d_l_eye.png" % int(file[0:-4]))):
            img_eye_left = cv2.imread(os.path.join(label_path, "%05d_l_eye.png" % int(file[0:-4])))
            img_eye_left = cv2.cvtColor(img_eye_left, cv2.COLOR_BGR2GRAY)
            img_eye_left = np.where(img_eye_left > 0, 3, 0)
            res.append(img_eye_left)

        if os.path.exists(os.path.join(label_path, "%05d_r_eye.png" % int(file[0:-4]))):
            img_eye_right = cv2.imread(os.path.join(label_path, "%05d_r_eye.png" % int(file[0:-4])))
            img_eye_right = cv2.cvtColor(img_eye_right, cv2.COLOR_BGR2GRAY)
            img_eye_right = np.where(img_eye_right > 0, 3, 0)
            res.append(img_eye_right)

        # 4.鼻子
        if os.path.exists(os.path.join(label_path, "%05d_nose.png" % int(file[0:-4]))):
            img_nose = cv2.imread(os.path.join(label_path, "%05d_nose.png" % int(file[0:-4])))
            img_nose = cv2.cvtColor(img_nose, cv2.COLOR_BGR2GRAY)
            img_nose = np.where(img_nose > 0, 4, 0)
            res.append(img_nose)

        # 5.上嘴唇
        if os.path.exists(os.path.join(label_path, "%05d_u_lip.png" % int(file[0:-4]))):
            img_upper_lip = cv2.imread(os.path.join(label_path, "%05d_u_lip.png" % int(file[0:-4])))
            img_upper_lip = cv2.cvtColor(img_upper_lip, cv2.COLOR_BGR2GRAY)
            img_upper_lip = np.where(img_upper_lip > 0, 5, 0)
            res.append(img_upper_lip)

        # 6.嘴巴
        if os.path.exists(os.path.join(label_path, "%05d_mouth.png" % int(file[0:-4]))):
            img_mouth = cv2.imread(os.path.join(label_path, "%05d_mouth.png" % int(file[0:-4])))
            img_mouth = cv2.cvtColor(img_mouth, cv2.COLOR_BGR2GRAY)
            img_mouth = np.where(img_mouth > 0, 6, 0)
            res.append(img_mouth)

        # 7.下嘴唇
        if os.path.exists(os.path.join(label_path, "%05d_l_lip.png" % int(file[0:-4]))):
            img_down_lip = cv2.imread(os.path.join(label_path, "%05d_l_lip.png" % int(file[0:-4])))
            img_down_lip = cv2.cvtColor(img_down_lip, cv2.COLOR_BGR2GRAY)
            img_down_lip = np.where(img_down_lip > 0, 7, 0)
            res.append(img_down_lip)

        res = np.array(res)
        res = res.max(0)
        # print(type(res), res.shape)

        res = res.reshape(h, w, 1)
        cv2.imwrite(os.path.join("F:/dataset/CelebAMask-HQ/labels", "%s.png" % file[0:-4]), res)

