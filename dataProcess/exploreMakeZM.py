import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


wt=1
dirPath=r'D:\YANG Luoxiao\Data\10台风机数据（20160205-20170605）\pickleF\PowerP8S1Train'
with open(dirPath,'rb') as f:
    data=pickle.load(f)[:,:,[0,7]]
#2-3w 4-5 w s1
# 4-5 w s4
blackName=[10,0,5]
dataR={}
for i in range(11):
    if i in blackName:
        continue
    k='ZMS1WT%d'%i
    dataU=np.concatenate((data[20000:30000,i,:],data[40000:50000,i,:]),axis=0)
    dataR[k]=dataU
    # plt.plot(dataU[:,0],dataU[:,1],'.')
    # plt.show()

dirPath=r'D:\YANG Luoxiao\Data\10台风机数据（20160205-20170605）\pickleF\PowerP8S4Train'
with open(dirPath,'rb') as f:
    data=pickle.load(f)[:,:,[0,7]]

for i in range(11):
    if i in blackName:
        continue
    k = 'ZMS4WT%d' % i
    dataU = data[40000:50000, i, :]
    dataR[k] = dataU

with open(r'D:\YANG Luoxiao\Data\WPC\ZMD','wb') as f:
    pickle.dump(dataR,f)














