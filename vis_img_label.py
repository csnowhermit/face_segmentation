import os
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.utils as vutils

import config

'''
    可视化图像及标签
'''

# 调色板
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(config.num_classes)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")


# 读取原图
if __name__ == '__main__':
    img_path = "F:/dataset/helen_face/images/"
    lbl_path = "F:/dataset/helen_face/labels/"
    for file in os.listdir(img_path):
        if os.path.exists(os.path.join(lbl_path, file[0:-4] + ".png")) is False:
            continue
        print(file)
        image = Image.open(os.path.join(img_path, file)).convert('RGB')
        label = Image.open(os.path.join(lbl_path, file[0:-4] + ".png")).convert('L')

        # 给label上色
        # r_label = Image.fromarray(label.byte().cpu().numpy())
        r_label = label
        r_label.putpalette(colors)
        r_label = r_label.convert('RGB')
        # print(type(r_label), np.array(r_label).shape)

        # 拼接：原图，标签
        show_img_list = [T.PILToTensor()(image) * 255., T.PILToTensor()(r_label) * 255.]

        label_show = vutils.make_grid(show_img_list, nrow=1, padding=2, normalize=True).cpu()

        vutils.save_image(label_show, os.path.join("F:/dataset/helen_face/test", "eval_%s" % file))
