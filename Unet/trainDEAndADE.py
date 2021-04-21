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
class WPCDEDataset(Dataset):
    def __init__(self, lenG, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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

# dirP=r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
dirP=r"D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE1\TrainData"
# saveFig = r'D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly\testRes'
saveFig = r'D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE1\testRes'
transf = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
setLen=4000
trainDatas=WPCDEDataset(setLen,dirP+r'\\'+'train',transform=transf)
valDatas=WPCDEDataset(int(setLen/4),dirP+r'\\'+'test',transform=transf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=16
lr = 2e-4
epochs = 20
weight_decay=0
minloss = 10
start_epoch=0
outf=r"D:\YANG Luoxiao\Model\WPC\DE\Unet"

unet = UNet(in_channels=3, out_channels=3)
unet.to(device)
optimizer = optim.Adam(list(unet.parameters()), lr=lr,weight_decay=weight_decay)


dataloaderT = torch.utils.data.DataLoader(trainDatas, batch_size=batch_size,
                                         shuffle=True, num_workers=int(0))

dataloaderV = torch.utils.data.DataLoader(valDatas, batch_size=batch_size,
                                         shuffle=True, num_workers=int(0))



for epoch in range(start_epoch, start_epoch + epochs):
    unet.train()


    for i, (x,y) in enumerate(dataloaderT):
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
    unet.eval()
    lossM=0
    c=0
    # with torch.no_grad():
    #     for i, (x, y) in enumerate(dataloaderV):
    #         c += 1
    #         x = x.to(device)
    #         y = y.to(device)
    #
    #         ypred = unet(x)
    #         loss = F.mse_loss(y, ypred)
    #         lossM+=loss
    #         # if epoch==198:
    #         #
    #         #     f, axs = plt.subplots(3, 3)
    #         #     for j in range(3):
    #         #         imgx = transform_invert(x[j, :].detach().cpu(), transf)
    #         #         axs[j, 0].imshow(imgx)
    #         #
    #         #         imgx = transform_invert(ypred[j, :].detach().cpu(), transf)
    #         #         axs[j, 1].imshow(imgx)
    #         #
    #         #         imgx = transform_invert(y[j, :].detach().cpu(), transf)
    #         #         axs[j, 2].imshow(imgx)
    #         #     f.savefig(saveFig + r'\\' + 'testepoch%dRes%d.jpg' % (epoch, c))
    #         #     plt.close(f)
    #     print('VAL loss= ', lossM / c)
    #     f, axs = plt.subplots(3, 3)
    #     for j in range(3):
    #         imgx = transform_invert(x[j, :].detach().cpu(), transf)
    #         axs[j, 0].imshow(imgx)
    #
    #         imgx = transform_invert(ypred[j, :].detach().cpu(), transf)
    #         axs[j, 1].imshow(imgx)
    #
    #         imgx = transform_invert(y[j, :].detach().cpu(), transf)
    #         axs[j, 2].imshow(imgx)
    #     f.savefig(saveFig + r'\\' + 'epoch%dRes%d.jpg' % (epoch, 0))
    #     plt.close(f)
    #
    #     # plt.show()

    state = {'model': unet.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    # torch.save(state, '%s/NoNormUnet800DE.pth' % (outf))
    torch.save(state, '%s/UnetS%dDEAndADEepoch%d.pth' % (outf,setLen,epoch))
    # if minloss > (lossM / c):
    #     state = {'model': unet.state_dict(), 'optimizer': optimizer.state_dict(),
    #              'epoch': epoch}
    #     # torch.save(state, '%s/NoNormUnet800DE.pth' % (outf))
    #     torch.save(state, '%s/UnetS%dDEAndADEBest.pth' % (outf,setLen))
    #     minloss = lossM / c
    #     # minmapeloss=lossm/ c
    #     print('bestloss:  ', minloss)































