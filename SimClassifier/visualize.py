import os
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

VISUAL_DIR = 'visual_act'
VISUAL_BAK = 'visual_others'
GLOBAL_offset = 0
img_path = ''


def visual_importance(model: nn.Module, trans, path_to_image):
    global GLOBAL_offset
    global img_path

    os.makedirs(VISUAL_DIR, exist_ok=True)
    os.makedirs(VISUAL_BAK, exist_ok=True)

    model.eval()
    for module in model.modules():
        module.register_forward_hook(plot_features)

    if isinstance(path_to_image, str):
        path_to_image = [path_to_image]
    else:
        assert isinstance(path_to_image, list)

    for file in path_to_image:
        print(file)
        GLOBAL_offset = 0
        img_path = file.split('/')[-1].split('.')[-2]
        os.makedirs(VISUAL_DIR + '/' + img_path, exist_ok=True)
        os.makedirs(VISUAL_BAK + '/' + img_path, exist_ok=True)
        try:
            img = Image.open(file)
        except:
            continue
        if len(img.size) == 2:
            img = img.convert("RGB")
        img = trans(img)
        batch = torch.unsqueeze(img, 0)
        out = model(batch)


def plot_features(self: nn.Module, input: torch.Tensor, output: torch.Tensor):
    global GLOBAL_offset

    size = output.size()
    if len(size) != 4:
        return

    dim_ch = size[1]        # pytorch default NCHW
    vis = output[0].detach().numpy()

    visual_dir = VISUAL_BAK
    if isinstance(self, nn.ReLU):
        visual_dir = VISUAL_DIR

    prefix = "{}/{}/{}_{}".format(visual_dir, img_path,
                                  self.__class__.__name__, GLOBAL_offset)

    GLOBAL_offset += 1

    num_col = 16 if dim_ch % 16 == 0 else 2
    num_row = dim_ch // 16
    H, W = size[2], size[3]
    out_vis = np.zeros((num_row*H, num_col*W))

    for row in range(num_row):
        for col in range(num_col):
            out_vis[row*H:(row+1)*H, col*W:(col+1)*W] += vis[row*num_col+col]

    out_vis -= np.min(out_vis)  # to [0, max]
    out_vis = out_vis/np.max(out_vis) * 255
    out_vis = out_vis.astype(np.uint8)

    img = Image.fromarray(out_vis)
    # img.convert('L')
    img.save(prefix+'.jpg')
    print("saved {}.jpg".format(prefix))
