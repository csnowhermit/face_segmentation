import os
import random
from PIL import Image
from torch.utils.data import Dataset

'''
    拆分数据集
'''
def split_dataset(img_path):
    trainset, valset = [], []
    for x in os.listdir(img_path):
        if random.randint(0, 10) <= 8:
            trainset.append(os.path.join(img_path, x))
        else:
            valset.append(os.path.join(img_path, x))
    return trainset, valset


class HelenFace(Dataset):
    def __init__(self, imglist, transform):
        self.transform = transform
        self.imglist = imglist    # 这里图像列表已经是全路径了
        self.annolist = [file.replace('helen', 'labels') for file in imglist]


    def __getitem__(self, index):
        img = Image.open(self.imglist[index]).convert("RGB")
        label = Image.open(self.annolist[index]).convert("RGB")

        return self.transform(img, label)

    def __len__(self):
        return len(self.imglist)

