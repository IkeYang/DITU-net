import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from model import UNet,transform_invert
import torch.nn.functional as F
import torch.optim as optim
def trainUnet(dirP,name,setLen,epochs=20):
    class WPCDEDataset(Dataset):
        def __init__(self, lenG, root_dir, transform=None):

            self.root_dir = root_dir
            self.lenG=lenG
            self.transform = transform

        def __len__(self):

            return self.lenG

        def __getitem__(self, idx):
            img_nameX =self.root_dir+'%dx.jpg'%(idx)
            img_nameY =self.root_dir+'%dy.jpg'%(idx)
            imageX = Image.open(img_nameX).convert('RGB')
            imageY = Image.open(img_nameY).convert('RGB')

            if self.transform:
                imageX = self.transform(imageX)
                imageY = self.transform(imageY)
            return imageX,imageY


    transf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainDatas=WPCDEDataset(setLen,dirP+r'\\'+'train',transform=transf)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=16
    lr = 2e-4
    weight_decay=0
    start_epoch=0
    outf=r"..\model"

    unet = UNet(in_channels=3, out_channels=3)
    unet.to(device)
    optimizer = optim.Adam(list(unet.parameters()), lr=lr,weight_decay=weight_decay)


    dataloaderT = torch.utils.data.DataLoader(trainDatas, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))

    # dataloaderV = torch.utils.data.DataLoader(valDatas, batch_size=batch_size,
    #                                          shuffle=True, num_workers=int(0))

    dataSplit=None # reserved rate of  data

    for epoch in range(start_epoch, start_epoch + epochs):
        unet.train()
        for i, (x,y) in enumerate(dataloaderT):
            if dataSplit is not None:
                if i > len(dataloaderT) * dataSplit:
                    break
            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()
            ypred = unet(x)
            loss=F.mse_loss(y,ypred)
            loss.backward()
            optimizer.step()
            # break
            if (i) % int(len(dataloaderT) / 4) == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t '
                      % (epoch, start_epoch + epochs, i, len(dataloaderT), loss))
        state = {'model': unet.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, '%s/UnetS%d%sepoch%d.pth' % (outf,setLen,name,epoch))































