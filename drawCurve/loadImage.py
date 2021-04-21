import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
dirP=r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
# img_nameX =dirP+r'\\'+'train1x.jpg'
# img_nameY =dirP+r'\\'+'train1y.jpg'
#
# imaX=io.imread(img_nameX)
# plt.imshow(imaX)
# imaX=io.imread(img_nameY)
# plt.imshow(imaX)
# plt.show()


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

transf = transforms.Compose(
    [

    transforms.ToTensor(),
    ])

trainDatas=WPCDEDataset(800,dirP+r'\\'+'train',transform=transf)

x,y=trainDatas[588]

transf = transforms.Compose(
    [
    transforms.Scale([572,572]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainDatas=WPCDEDataset(800,dirP+r'\\'+'train',transform=transf)

x,y=trainDatas[588]
print('a')



















