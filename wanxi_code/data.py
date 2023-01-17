import os
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *

# Convert a PIL Image to tensor.
transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'train_masks'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        msk_name = self.name[index]
        msk_path = os.path.join(self.path, 'train_masks', msk_name)
        img_path = os.path.join(self.path, 'train_images', msk_name.replace('png', 'jpg'))
        msk = keep_mask_size_open(msk_path)
        img = keep_image_size_open(img_path)
        return transform(img), transform(msk)

if __name__ == '__main__':
    data = MyDataset('')
    print(data[0][0].shape)
    print(data[0][1].shape)
