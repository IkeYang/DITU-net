import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def addData(data,ind,thres,var=0.01):
    lenD=len(ind)
    bs = int(thres/lenD)
    res=np.zeros([bs*lenD,2])
    x=data[ind,0]
    y=data[ind,1]
    for i in range(bs):
        res[lenD*i:lenD*(i+1),0]=x+np.random.randn(lenD)*var
        res[lenD*i:lenD*(i+1),1]=y+np.random.randn(lenD)*var
    res[np.where(res > 1)] = 1
    res[np.where(res < 0)] = 0
    return res
def checkAndEnhancYregion(data,deviation=10):
    y = data[:, 1]
    lenD=len(y)
    thres = 0.05 * 10000*10/deviation
    for i in range(deviation):

        start=1/deviation*i
        end=1/deviation*(i+1)
        # p=np.sum((y> start) * (y<end))/lenD
        # print(p)
        ind=np.where(((y> start) * (y<end))==1)[0]
        if len(ind)==0:
            continue
        #
        # print(len(ind)/lenD)
        if len(ind)<thres:
            # print(len(ind))
            data=np.vstack((data,addData(data, ind, thres)))
    return data

if __name__ == '__main__':
    path = r'D:\YANG Luoxiao\Data\WPC\newRes'
    with open(path, 'rb') as f:
        newRes = pickle.load(f)
    # data=newRes['data']
    # thresRes=newRes['thres']

    # data={}
    # thresRes={}
    # data['JF-78']=newRes['data']['JF-78']
    data = newRes['data']['JF-78']
    # data=newRes['data']['JF-33']
    # thresRes['JF-78']=newRes['thres']['JF-78']
    print(len(data))## 9510
    # data=checkAndEnhancYregion(data)
    f,axs=plt.subplots(2,2)
    # axs[0,0].plot(data[:,0],data[:,1],'*')
    axs[0,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(x,y,'*')
    # plt.show()

    data=checkAndEnhancYregion(data,deviation=5)
    print(len(data))## 9510
    # f,axs=plt.subplots(1,1)
    # axs[0,0].plot(data[:,0],data[:,1],'*')
    axs[0,1].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(x,y,'*')
    # plt.show()



    data=checkAndEnhancYregion(data,deviation=10)
    print(len(data))## 9510
    # f,axs=plt.subplots(1,1)
    # axs[0,0].plot(data[:,0],data[:,1],'*')
    axs[1,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(x,y,'*')
    # plt.show()


    data=checkAndEnhancYregion(data,deviation=30)
    print(len(data))## 9510
    # f,axs=plt.subplots(1,1)
    # axs[0,0].plot(data[:,0],data[:,1],'*')
    axs[1,1].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(data[:,0],data[:,1],'*')
    # axs[1,0].plot(x,y,'*')
    plt.show()