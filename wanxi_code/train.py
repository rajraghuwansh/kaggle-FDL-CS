from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'unet_param/unet.pth'
data_path = ''
save_path = 'trained_images'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCEWithLogitsLoss()

    epoch = 1

    while True:
        for i, (img, msk) in enumerate(data_loader):  # Create dataset
            img, msk = img.to(device), msk.to(device)  # Put images to our device

            out_img = net(img)
            train_loss = loss_fun(out_img, msk)

            opt.zero_grad()  # Clear records
            train_loss.backward()
            opt.step  # Clear records

            # Print train loss every 5 epochs
            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # Save weight every 50 epochs
            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            # Visualize the output
            _img = img[0]
            _msk = msk[0]
            _out_img = out_img[0]

            image = torch.stack([_img, _msk, _out_img], dim=0)  # Stack 3 images
            save_image(image, f'{save_path}/{i}.png')

        epoch += 1