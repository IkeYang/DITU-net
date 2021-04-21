#Author:ike yang
from testPic import pipLineTestUnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def findBestParam(data,thresRes,name,repeatTimes=10):
    resM=10
    for epoch in range(1,15):
        for ms in [0.6,0.8,0.9,1,1.2,1.4,1.6,1.8,2]:
            print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),4000))
            res = []
            for i in range(repeatTimes):
                seq, ret, res1, res2, res3,_ = pipLineTestUnet(data, thresRes, epoch=epoch, ms=ms, name=name,
                                                             setLen=4000, bs=bs)
                res.append(np.mean(res1))
            res=np.mean(res)
            print(res)
            if res<resM:
                resM=res
                print('progress best until now')
                print(res)
                bestEpoch=epoch
                bestms=ms
    return bestEpoch,bestms


path = r'D:\YANG Luoxiao\Data\WPC\newRes'
with open(path, 'rb') as f:
    newRes = pickle.load(f)
data=newRes['data']
thresRes=newRes['thres']


# epoch=3
# ms=0.9
bs=20

# res={}
with open('paraVP', 'rb') as f:
    res=pickle.load(f)
# for vp in range(9,10):
#     name = 'DEAndADEVP%d' % (vp)
#     if vp==0:
#         name = 'DEAndADE'
#     print('********************************')
#     bestEpoch,bestms=findBestParam(data,thresRes,name,repeatTimes=5)
#     res[name]=(bestEpoch,bestms)
# with open('paraVP', 'wb') as f:
#     pickle.dump(res, f)

for (k,v) in res.items():
    name=k
    epoch,ms=v
    print("name: %s epoch: %d ms: %s sLen: %d "%(name,epoch,str(ms),4000))
    seq, ret, res1, res2, res3, _ = pipLineTestUnet(data, thresRes, epoch=epoch, ms=ms, name=name,
                                                    setLen=4000, bs=bs)
    print(np.mean(res1))







