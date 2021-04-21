#Author:ike yang
from testPic import pipLineTestUnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
path = r'D:\YANG Luoxiao\Data\WPC\newRes'
with open(path, 'rb') as f:
    newRes = pickle.load(f)
data=newRes['data']
thresRes=newRes['thres']



def findBestParam(data,thresRes,name,repeatTimes=10):
    resM=10
    for epoch in range(15):
        for ms in [0.6,0.8,0.9,1,1.2,1.4,1.6,1.8,2]:
            print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),4000))
            res = []
            for i in range(repeatTimes):
                seq, ret, res1, res2, res3,_ = pipLineTestUnet(data, thresRes, epoch=epoch, ms=ms, name=name,
                                                             setLen=4000, bs=bs)
                res.append(np.mean(res1))
            res=np.mean(res)
            if res<resM:
                resM=res
                print('progress best until now')
                print("name: %s epoch: %d ms: %s sLen: %d " % (name, epoch, str(ms), 4000))
                bestEpoch=epoch
                bestms=ms
    return bestEpoch,bestms





# epoch=3
# ms=0.9
# bs=8
# for setLen in [2000,3000,4000,5000,6000,7000,8000]:
#     name='DEAndADE'
#     seq,ret,res1,res2,res3=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs)
#     print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
#     print (np.mean(res1))
#     print (np.mean(res2))
#     print (np.mean(res3))

# data={}
# thresRes={}
# data['JF-78']=newRes['data']['JF-78']
# thresRes['JF-78']=newRes['thres']['JF-78']

epoch=3
ms=0.9
bs=8
setLen=4000

# name='DEAndADE'
# res=[]
# seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs,enhance=True,dataType='Ndefault')
# seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs,enhance=False,dataType='Ndefault')
# # print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
# # res.append(np.mean(res1))
# print (np.mean(res1))


epoch=3
ms=0.9
bs=8
setLen=4000

name='DEAndADE'
res=[]
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.85*100)
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.875*100)
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.90*100)
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.925*100)
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.95*100)
seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=0.8,name=name,setLen = setLen,bs=bs,
                                         enhance=True,dataType='Ndefault',cutdownX=True,xPer=0.975*100)
# print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
# res.append(np.mean(res1))
print (np.mean(res1))















# x=np.arange(0,100)/100
# seq,ret,res1,res2,res3,(f1,f2,f3)=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs)
# seq,ret,res1,res2,res3,(f12,f2,f3)=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs,enhance=True)
# # f,axs=plt.subplots(1,2)
# #
# # axs[0].plot(data['JF-78'][:,0],data['JF-78'][:,1],'*')
# # axs[0].plot(x,f1[0](x))
# # axs[1].plot(x,y,'*')
# # plt.show()
# plt.plot(data['JF-78'][:,0],data['JF-78'][:,1],'*')
# plt.plot(x,f1[0](x))
# plt.plot(x,f12[0](x))
# plt.show()
# for epoch in range(3,4):
#     for ms in [0.9]:
#         print(epoch,ms)
#         print('Enhance False')
#         seq,ret,res1,res2,res3,_=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs,enhance=False)
#         # print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
#         print (np.mean(res1))
#         print('Enhance True')
#         seq, ret, res1, res2, res3, _ = pipLineTestUnet(data, thresRes, epoch=epoch, ms=ms, name=name, setLen=setLen,
#                                                         bs=bs,enhance=True)
#         # print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
#         print(np.mean(res1))
#         # print (np.mean(res2))
#         # print (np.mean(res3))

# res=[]
# for i in range(10):
#     seq,ret,res1,res2,res3=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs)
#     print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
#     res.append(np.mean(res1))
# print (np.mean(res))
    # print (np.mean(res2))
    # print (np.mean(res3))

# res=[]
# for i in range(10):
#     seq,ret,res1,res2,res3=pipLineTestUnet(data,thresRes,epoch=epoch,ms=ms,name=name,setLen = setLen,bs=bs,enhance=True)
#     print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),setLen))
#     res.append(np.mean(res1))
# print (np.mean(res))














