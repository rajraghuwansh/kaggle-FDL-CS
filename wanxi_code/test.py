import os
import torch

from net import *
from utils import keep_image_size_open, keep_mask_size_open
from data import *
from torchvision.utils import save_image

net = UNet().cpu()

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('sucessfully')
else:
    print('no loading')

result_path = 'result'
dir = 'test_images/'
for index, i in enumerate(os.listdir(dir)):
    img = keep_image_size_open(dir + i)
    img_data = transform(img).cpu()
    img_data = torch.unsqueeze(img_data, dim=0)

    out = net(img_data)
    save_image(out, f'{result_path}/{i}.png')
