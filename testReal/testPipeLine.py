import pickle
import sys
sys.path.append(r"..")
# sys.path.append(r"..\..")
from Unet.testPic import pipLineTestUnet
from dataProcess.clearWithCoord import fliterData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from utlize import delSomeName
warnings.filterwarnings('ignore')

path = r'D:\YANG Luoxiao\Data\WPC\newRes'
with open(path, 'rb') as f:
    newRes = pickle.load(f)
data = newRes['data']
thresRes = newRes['thres']
# delNameL=['H1-25F','A1-02','LN49','LN54']
delNameL=['H1-25F','A1-02','LN49',]
delete=True

if delete:
    compareRes = np.zeros([len(thresRes.keys())-len(delNameL), 2, 3])

else:
    compareRes = np.zeros([len(thresRes.keys()), 2, 3])
#delete=True
# [[0.05376273 0.05374136 0.05374893]
#  [0.059129   0.05687785 0.05567048]]
# [[0.0850055  0.08504637 0.08465071]
#  [0.0959916  0.09217918 0.08433428]]
#delete=False
# [[0.05381016 0.05378988 0.05379903]
#  [0.05639191 0.05414939 0.05289613]]
# [[0.08621667 0.08625056 0.08588949]
#  [0.09717228 0.09321366 0.08583511]]


bs = 20
setLen = 4000
epoch=3
ms=0.9

seq,rate,res1,res2,res3,(f1,f2,f3)= pipLineTestUnet(data,thresRes,epoch=3,ms=ms,name='DEAndADE',setLen = setLen,bs=bs,enhance=True,dev=10,testOnly=False,dataType='default',saveMiddle=False)
seq,l=delSomeName(seq,delNameL=delNameL,delete=delete)
print(seq)
compareRes[:, 0, 0] = res1[l]
compareRes[:, 0, 1] = res2[l]
compareRes[:, 0, 2] = res3[l]
rate=rate[l,:]



from testReal.testBaseline import testSpline

for i,k in enumerate(seq):
    data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True, minmax=False)
    # data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True, minmax=True)
    r1, r2, r3 = testSpline(data1, data2, data3, dT)
    compareRes[i,1,0]=r1
    compareRes[i,1,1]=r2
    compareRes[i,1,2]=r3







improvedZM1=compareRes[:,0,0]-compareRes[:,1,0]
improvedZM2=compareRes[:,0,1]-compareRes[:,1,1]
improvedZM3=compareRes[:,0,2]-compareRes[:,1,2]

f,axs=plt.subplots(2,3)
axs[0,0].plot(rate[:,0],improvedZM1,'*')
axs[0,1].plot(rate[:,0],improvedZM2,'*')
axs[0,2].plot(rate[:,0],improvedZM3,'*')

axs[1,0].plot(rate[:,1],improvedZM1,'*')
axs[1,1].plot(rate[:,1],improvedZM2,'*')
axs[1,2].plot(rate[:,1],improvedZM3,'*')
plt.show()
extract=compareRes[:,0,:]-compareRes[:,1,:]
print(1)





ZMr1r2=[]
ZMR1r2=[]
ZMR1R2=[]
ZMr1R2=[]
zmr1=0.985
zmr2=0.9
zmInd1=[]
zmInd2=[]
zmInd3=[]
for (i,k) in enumerate(seq):
    r1=rate[i,0]
    r2=rate[i,1]
    if r1>zmr1:
        if r2>zmr2:
            ZMR1R2.append(k)
            zmInd1.append(i)
        else:
            ZMR1r2.append(k)
    else:
        if r2 > zmr2:
            ZMr1R2.append(k)
            zmInd1.append(i)
        else:
            ZMr1r2.append(k)
            zmInd2.append(i)
print('both bigger',ZMR1R2)
print('r1 bigger',ZMR1r2)
print('r2 bigger',ZMr1R2)
print('both smaller',ZMr1r2)


print(np.mean(compareRes[zmInd1,:,:],axis=0))
print(np.mean(compareRes[zmInd2,:,:],axis=0))
