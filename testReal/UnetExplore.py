import sys
import pickle
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"C:\MyPhDCde\我的坚果云\windPowerCurveModeling")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataProcess.clearWithCoord import fliterData
from testReal.testUnet import testUnet


if __name__ == '__main__':

    path = r'D:\YANG Luoxiao\Data\WPC\WPC'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
        thresRes = pickle.load(f)
    i=1
    for (k,v) in thresRes.items():
        if 'H1-13F' in k:  #H1-14F H1-03F A2-22
            data1,data2,data3,dT=fliterData(data[k],thresRes[k],trainTestSPlit=True)
            r1, r2, r3 = testUnet(data1, data2, data3, dT, k, epoch=0,ms=0.8)
            print(r1 / i, r2 / i, r3 / i)
            r1, r2, r3 = testUnet(data1, data2, data3, dT, k, epoch=4,ms=0.5)
            print(r1 / i, r2 / i, r3 / i)
            r1, r2, r3 = testUnet(data1, data2, data3, dT, k, epoch=1)
            print(r1 / i, r2 / i, r3 / i)










