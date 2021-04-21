import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pickle
import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"..")
from Unet.testPic import dataToImageBench,benchTest,benchImageTest,testPict
from Unet.openImageNp import imageToLine
from sklearn.metrics import mean_squared_error
from dataProcess.clearWithCoord import fliterData
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
import warnings
warnings.filterwarnings('ignore')



def testUnet(data1,data2,data3,dataT,name,epoch=1,ms=None):
    '''
    
    :param data1: orginal Data
    :param data2: processed Data
    :param data3: processed Data
    :param dataT: test Data
    :param name: 
    :return: 
    '''

    y1, scaler = testPict(data1, name + 'data1', path=r'D:\YANG Luoxiao\Data\gearbox', orginal=True,plot=False,epoch=epoch,ms=ms)
    if data2 is None:
        xi, yi, f1 = imageToLine(y1)
        x1P = f1(dataT[:, 0])
        return x1P
    y2,_=testPict(data2, name + 'data2', path=r'D:\YANG Luoxiao\Data\gearbox', orginal=False, scaler=scaler,plot=False,epoch=epoch,ms=ms)
    y3,_=testPict(data3, name + 'data3', path=r'D:\YANG Luoxiao\Data\gearbox', orginal=False, scaler=scaler,plot=False,epoch=epoch,ms=ms)

    xi, yi, f1 = imageToLine(y1)
    xi, yi, f2 = imageToLine(y2)
    xi, yi, f3 = imageToLine(y3)
    x1P=f1(dataT[:,0])
    x2P=f2(dataT[:,0])
    x3P=f3(dataT[:,0])

    return np.sqrt(mean_squared_error(x1P, dataT[:,1])),np.sqrt(mean_squared_error(x2P, dataT[:,1])),np.sqrt(mean_squared_error(x3P, dataT[:,1]))









if __name__=='__main__':
    #
    # path = r'D:\YANG Luoxiao\Data\WPC\WPC'
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    #     thresRes = pickle.load(f)
    #
    # minL=10
    # for i in range(1000):
    #     for epoch in [3]:
    #         for ms in [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]:
    #             # print(epoch,ms)
    #             data1Torch,data2Torch,data3Torch,dataTTorch,dtL=dataToImageBench(data, thresRes, ms=ms)
    #             yR = benchTest(data1Torch, epoch=epoch, bs=16)
    #             res = np.mean(benchImageTest(yR, dtL))
    #             # print(res)
    #             if res<minL:
    #                 minL=res
    #                 bepoch=epoch
    #                 bms=ms
    #                 print('BestPara',bepoch,bms,'loss',minL)

      # BestPara 3 0.9 loss 0.07307141453345091  dataset1





    # path = r'D:\YANG Luoxiao\Data\WPC\ZMD'
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'rb') as f:
    #     thresRes = pickle.load(f)
    #
    # minL=10
    # for i in range(1000):
    #     for epoch in range(21):
    #         for ms in [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]:
    #             # print(epoch,ms)
    #             data1Torch,data2Torch,data3Torch,dataTTorch,dtL,_=dataToImageBench(data, thresRes, ms=ms,minmax=False,seq=True)
    #             yR = benchTest(data1Torch, epoch=epoch, bs=16)
    #             res = np.mean(benchImageTest(yR, dtL))
    #             # print(res)
    #             if res<minL:
    #                 minL=res
    #                 bepoch=epoch
    #                 bms=ms
    #                 print('BestPara',bepoch,bms,'loss',minL)
    #     #BestPara 3 1.8 loss 0.05921329441469157







    #
    #
    #
    #

    path = r'D:\YANG Luoxiao\Data\WPC\newRes'
    with open(path, 'rb') as f:
        newRes = pickle.load(f)
    data=newRes['data']
    thresRes=newRes['thres']
    compareRes = np.zeros([len(thresRes.keys()),2,3])
    minL=10
    bs=8
    setLen=4000
    for epoch in [3]:
        for ms in [0.9]:#zm 1.8 10.9
            print(epoch,ms)
            data1Torch,data2Torch,data3Torch,dataTTorch,dtL,seq,rate=dataToImageBench(data, thresRes, ms=ms,minmax=False,seq=True,rateR=True)
            # data1Torch,data2Torch,data3Torch,dataTTorch,dtL,seq,rate=dataToImageBench(data, thresRes, ms=ms,minmax=True,seq=True,rateR=True)
            print(seq)
            yR = benchTest(data1Torch, epoch=epoch, bs=bs,setLen=setLen)
            res=benchImageTest(yR, dtL)
            compareRes[:,0,0]=res
            print(res)
            res = np.mean(res)
            yR = benchTest(data2Torch, epoch=epoch, bs=bs)
            res = benchImageTest(yR, dtL)
            compareRes[:, 0, 1] = res
            print(res)
            res = np.mean(res)
            yR = benchTest(data3Torch, epoch=epoch, bs=bs)
            res = benchImageTest(yR, dtL)
            compareRes[:, 0, 2] = res
            print(res)
            res = np.mean(res)
            if res<minL:
                minL=res
                bepoch=epoch
                bms=ms
                print('BestPara',bepoch,bms,'loss',minL)

      # BestPara 3 0.9 loss 0.07307141453345091




    from testReal.testBaseline import testSpline
    i=0
    for k in seq:
        data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True, minmax=False)
        # data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True, minmax=True)
        r1, r2, r3 = testSpline(data1, data2, data3, dT)
        compareRes[i,1,0]=r1
        compareRes[i,1,1]=r2
        compareRes[i,1,2]=r3
        i+=1
        print(k, r1, r2, r3)
    print(compareRes)
    # for  i in range(21):
    # data = compareRes[:, :, 0]
    # plt.plot(data[:, 0])
    # plt.plot(data[:, 1])
    # plt.show()
    # data = compareRes[:, :, 1]
    # plt.plot(data[:, 0])
    # plt.plot(data[:, 1])
    # plt.show()
    # data = compareRes[:, :, 2]
    # plt.plot(data[:, 0])
    # plt.plot(data[:, 1])
    # plt.show()

    with open('resNew','wb') as f:
        pickle.dump((compareRes,seq,rate),f)

    print(np.mean(compareRes[:,0,:],axis=0))
    print(np.mean(compareRes[:,1,:],axis=0))





















