import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
import os

def returnData(path,filename):
    path1 = path + r'\\' + filename
    if 'A' in filename and '-' in filename:
        # continue
        variable = ['风速', '功率']
        print(filename)
        dataF = pd.read_csv(path1, encoding='gbk', sep=',')
    # elif filename == 'JF-33.csv' or filename == 'JF-50.csv' or 'LN' in filename:
    #     variable = ['wind_speed_2', 'nacelle_temperature']  # shift 1
    #     print(filename)
    #     dataF = pd.read_csv(path1, encoding='gbk', sep=',')

    elif 'JF' in filename or 'LN' in filename:
        variable = ['wind_speed', 'grid_power']
        print(filename)
        dataF = pd.read_csv(path1, encoding='gbk', sep=',',index_col=False)
    elif 'H' in filename:
        variable = ['风速', '功率']
        print(filename)
        dataF = pd.read_excel(path1)
    data = pd.DataFrame()
    for i in variable:
        data[i] = dataF[i]
    data = data.values
    return data
# JF=['JF-33','JF-50','JF-64','JF-78']
# LN=["LN35","LN40","LN41","LN42","LN43","LN49","LN51","LN53","LN54","LN62","LN63"]
# GJG1=["A1-0"]
# GJG2=["A1-0"]



res={}

path=r'D:\YANG Luoxiao\Data\WPC\Excel1'
L=os.listdir(path)
L=sorted(L)
print(len(L))
for filename in L:
    data=returnData(path,filename)
    name,_=filename.split('.')
    # plt.plot(data[:,0],data[:,1],'.')
    # plt.show()
    res[name]=data
with open(r'D:\YANG Luoxiao\Data\WPC\res','wb') as f:
    pickle.dump(res,f)



path=r'D:\YANG Luoxiao\Data\WPC\Excel2'
L=os.listdir(path)
L=sorted(L)
print(len(L))
for filename in L:
    data=returnData(path,filename)
    name,_=filename.split('.')
    # plt.plot(data[:,0],data[:,1],'.')
    # plt.show()
    res[name]=data
with open(r'D:\YANG Luoxiao\Data\WPC\res','wb') as f:
    pickle.dump(res,f)















































