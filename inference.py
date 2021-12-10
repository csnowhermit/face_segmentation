import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms as T
import torchvision.utils as vutils

import config
from model import load_model

'''
    模型推理过程
'''

def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 这里使用backbone+deeplabv3+。推理时直接写实际的num_classes即可
    if config.backbone == 'resnet50':
        model = load_model('resnet50', num_classes=config.num_classes, output_stride=config.output_stride)
    else:
        model = load_model('mobilenetv2', num_classes=config.num_classes, output_stride=config.output_stride)

    # 加载预训练模型
    if len(config.pretrained_model) > 0 and os.path.isfile(config.pretrained_model):
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)

        model.load_state_dict(checkpoint["model_state"])
        if config.use_gpu and config.num_gpu > 1:
            model = nn.DataParallel(model)
        model.to(config.device)
        del checkpoint
    else:
        print("Please check your pretrained_modoel...")
        return

    if config.use_gpu and torch.cuda.is_available():
        model.cuda()    # 手动将model放到cuda上

    model.eval()
    for file in os.listdir(config.inference_imgpath):
        print("Processing... %s" % os.path.join(config.inference_imgpath, file))

        img = Image.open(os.path.join(config.inference_imgpath, file)).convert("RGB")
        img = torch.tensor(T.PILToTensor()(img).unsqueeze(0), dtype=torch.float32)
        img = F.normalize(img, mean, std)  # 先做成BCHW的，再标准化
        h, w = img.shape[2], img.shape[3]
        img = img.to(config.device, dtype=torch.float32)  # BCHW的
        preds = model(img)

        outputs = preds.max(1)[1][0]

        # 调色板
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(config.num_classes)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # 给推理结果上色
        r_output = Image.fromarray(outputs.byte().cpu().numpy()).resize((w, h))  # resize时应该写wh
        r_output.putpalette(colors)
        r_output = r_output.convert('RGB')

        # 拼接：原图、推理结果
        # show_img_list = [img[0] * 255., T.PILToTensor()(r_output) * 255.]
        show_img_list = [img.detach().cpu()[0] * 255., T.PILToTensor()(r_output)]

        result_show = vutils.make_grid(show_img_list, nrow=1, padding=2, normalize=True).cpu()

        if os.path.exists(config.val_results_path) is False:
            os.makedirs(config.val_results_path)

        vutils.save_image(result_show, os.path.join(config.val_results_path, "inference_%s.png" % file[0:-4]))
        del result_show
        del outputs
        del img


if __name__ == '__main__':
    main()
