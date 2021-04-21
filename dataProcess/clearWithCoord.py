import numpy as np
import sys
# sys.path.insert(7,r"c:\users\user\anaconda3\lib\site-packages")
print(sys.path)
# import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

def deleteTest(data,dataTest):
    dataS=np.sum(data,axis=-1)
    dataST=np.sum(dataTest,axis=-1)
    lenD=~np.isin(dataS,dataST)
    return data[lenD,:]




def fliterData(data,thres,trainTestSPlit=False,splitP=0.2,minmax=True,xcutDown=False,xpercentilep=0.9,drawModel=False,preTrain=False,deleteNum=1):
    import random
    np.random.seed(1)
    random.seed(1)
    if minmax:
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        thres=min_max_scaler.transform(thres)
    dataNanInd=np.sum(np.isnan(data),axis=-1)>0
    data=data[~dataNanInd,:]

    if drawModel and  not preTrain:
        for ll in range(deleteNum):
            data=np.delete(data, np.argmax(data[:, 1]), axis=0)
            data=np.delete(data, np.argmax(data[:, 0]), axis=0)
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        thres = min_max_scaler.transform(thres)






    # number = len(np.where(data[:, 1] > 0.99)[0])
    # # print(number)
    # if number < 10:
    #     data[np.where(data[:, 1] > 0.99), :] = 0
    #     data[:, 1] = data[:, 1] / np.max(data[:, 1])



    # no process
    data1=data

    # process 0
    ind=np.where((data[:,0]>0.3 ))
    ind2=np.where((data[:,1]<0.04 ))
    ind=list(set(ind[0]) & set(ind2[0]))
    data2=np.delete(data, ind, 0)
    xThres = thres[0, 0]
    yThres = thres[0, 1]

    ind = np.where((data2[:, 0] > xThres))
    ind2 = np.where((data2[:, 1] < yThres))
    ind = list(set(ind[0]) & set(ind2[0]))
    data2 = np.delete(data2, ind, 0)
    data3=np.copy(data2)
    
    
    #totaly process
    lenT=thres.shape[0]
    for i in range(lenT):
        xThres=thres[i,0]
        yThres=thres[i,1]

        ind = np.where((data3[:, 0] > xThres))
        ind2 = np.where((data3[:, 1] < yThres))
        ind = list(set(ind[0]) & set(ind2[0]))
        data3 = np.delete(data3, ind, 0)
    if not trainTestSPlit:
        if xcutDown:
            num = np.percentile(data[:, 0], xpercentilep)
            data1Ind=data1[:,0]<num
            data1=data1[data1Ind,:]
            data1Ind = data2[:, 0] < num
            data2 = data2[data1Ind, :]
            data1Ind = data3[:, 0] < num
            data3 = data3[data1Ind, :]
        return data1,data2,data3
    else:
        testInd=np.random.choice(np.arange(0,data3.shape[0]),replace=False,size=int(data3.shape[0]*splitP))
        dataTest=data3[testInd,:]
        deleteTest(data1, dataTest)
        deleteTest(data2, dataTest)
        deleteTest(data3, dataTest)
        if xcutDown:
            num = np.percentile(data[:, 0], xpercentilep)
            data1Ind=data1[:,0]<num
            data1=data1[data1Ind,:]
            data1Ind = data2[:, 0] < num
            data2 = data2[data1Ind, :]
            data1Ind = data3[:, 0] < num
            data3 = data3[data1Ind, :]
        return data1,data2,data3,dataTest

if __name__=='__main__':

    with open(r'D:\YANG Luoxiao\Data\WPC\res','rb') as f:
        res=pickle.load(f)
    # with open(r'D:\YANG Luoxiao\Data\WPC\ZMD','rb') as f:
    #     res=pickle.load(f)
    with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'rb') as f:
        thresRes=pickle.load(f)


    for (k,v) in res.items():
        if k=='LN49':
            data1,data2,data3,dT=fliterData(res[k],thresRes[k],trainTestSPlit=True)
            plt.plot(data1[:, 0], data1[:, 1], '.',color='black')
            plt.plot(data2[:, 0], data2[:, 1], '.',color='blue')
            plt.plot(data3[:, 0], data3[:, 1], '.',color='yellow')
            plt.plot(dT[:, 0], dT[:, 1], '.',color='red')
            plt.show()

    with open(r'D:\NutCLoud\Nutstore\windPowerCurveModeling\testReal\resZM', 'rb') as f:
        r2 = pickle.load(f)

    with open(r'D:\NutCLoud\Nutstore\windPowerCurveModeling\testReal\res1', 'rb') as f:
        r1 = pickle.load(f)
    resC1 = r1[0][:, 0, :] - r1[0][:, 1, :]

    resC2=r2[0][:,0,:]-r2[0][:,1,:]






