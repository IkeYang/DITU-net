import os
import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"C:\MyPhDCde\我的坚果云\windPowerCurveModeling")
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from Unet.model import UNet,transform_invert
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataProcess.clearWithCoord import fliterData
path = r'D:\YANG Luoxiao\Data\WPC\WPC'
with open(path, 'rb') as f:
    data = pickle.load(f)
with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    thresRes = pickle.load(f)


def plotPict(data,name):
    pass




for (k, v) in thresRes.items():

    data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True)
    


















































