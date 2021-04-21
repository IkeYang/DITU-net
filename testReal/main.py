import sys
import pickle
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"..")
import numpy as np
import random
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from dataProcess.clearWithCoord import fliterData
from Unet.testPic import pipLineTestUnet
# from testReal.testBaseline import testDE
# from testReal.testBaseline import testSVR
from testReal.testBaseline import testSNN
from testReal.testBaseline import testSNN2
from testReal.testBaseline import testKNN
from testReal.testBaseline import testSpline
from baseline.testCurveModel import testCurveModel
from utlize import delSomeName
def MAPELoss(output, target):
    output=output.flatten()
    target=target.flatten()
    return np.mean(np.abs((target - output) / (target+1e-12)))
from sklearn.metrics import mean_squared_error,mean_absolute_error
def RMSE(output, target):
    output=output.flatten()
    target=target.flatten()
    return np.sqrt(mean_squared_error(output,target))
def MAE(output, target):
    output=output.flatten()
    target=target.flatten()
    return mean_absolute_error(output,target)
def reserveList(fL,l):
    f=[]
    for i in l:
        f.append(fL[i])
    return f
def testOnce(cutdownX = True,xPer = 0.85 * 100,dataType='default',domainCorrlation=False,method='ALL',minmax=True,nP=200):


    path = r"D:\YANG Luoxiao\Data\WPC\newRes"
    # path = r"D:\YANG Luoxiao\Data\WPC\newRes"
    with open(path, "rb") as f:
        newRes = pickle.load(f)
    data = newRes["data"]
    if minmax:
        for key, value in data.items():
            # number = len(np.where(value[:, 1] > 0.99)[0])
            # # print(number)
            # if number < 10:
            #     value[np.where(value[:, 1] > 0.99), :] = 0
            value = value[~np.isnan(value[:, 0]), :]
            value = value[~np.isnan(value[:, 1]), :]
            data[key] = value / np.max(value, axis=0)[1]

    thresRes = newRes["thres"]

    timeCost=0
    ##test Unet
    delNameL = ['H1-30F','H1-25F', 'A1-02', 'LN49', 'LN41','LN35','A1-08','LN63']
    delNameL = ['LN41','LN35','A1-08','LN63']
    delNameL = ['A1-02','ZMS4WT1','A2-25','ZMS1WT9','A1-10']
    delNameL = ['A1-02','A2-25','ZMS1WT9','A1-10']
    delete = True

    if delete:
        compareRes = np.zeros([len(thresRes.keys()) - len(delNameL), 2, 3])

    else:
        compareRes = np.zeros([len(thresRes.keys()), 2, 3])

    bs = 20
    setLen = 4000
    epoch = 3
    ms = 0.8
    name='DEAndADE'
    # nP
    # dataType='Nondefault'
    seq, rate, res1, res2, res3, (f1, f2, f3) = pipLineTestUnet(data, thresRes, epoch=3, ms=ms, name=name,
                                                                setLen=setLen, bs=bs, enhance=True, dev=10,
                                                                testOnly=False, dataType=dataType, saveMiddle=False,
                                                                cutdownX=cutdownX,xPer=xPer,domainCorrlation=domainCorrlation,nP=nP)
    # if cutdownX:
    #     with open('FuncUnetCTotal0-8' + str(xPer), 'wb') as f:
    #         pickle.dump((f1, f2, f3), f)
    # else:
    #     with open('FuncUnetTotal0-8', 'wb') as f:
    #         pickle.dump((f1, f2, f3), f)
    # return 0
    # if delete:
    seq, l = delSomeName(seq, delNameL=delNameL, delete=delete)
    res = {}
    res['seq']=seq
    res['rate']=rate[l]
    res['methodOrder']=['Unet','KNN','SNN','DE','ADE','PLF4','PLF5','Spline']
    res['res']=np.zeros([len(res['methodOrder']),len(seq),3,3]) # method , wt, 3dataCleanType,2 metric rmse mape MAE
    f1Unet=reserveList(f1,l);f2Unet=reserveList(f2,l);f3Unet=reserveList(f3,l)
    f1KNN=[];f2KNN=[];f3KNN=[]
    f1SNN=[];f2SNN=[];f3SNN=[]
    f1DE=[];f2DE=[];f3DE=[]
    f1ADE=[];f2ADE=[];f3ADE=[]
    f1PLF4=[];f2PLF4=[];f3PLF4=[]
    f1PLF5=[];f2PLF5=[];f3PLF5=[]
    f1Spline=[];f2Spline=[];f3Spline=[]





    for i,k in enumerate(seq):

        data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True,minmax=False,xcutDown=cutdownX,xpercentilep=xPer)
        #Unet
        res['res'][0,i,0,0]=RMSE(f1Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,0,1]=MAPELoss(f1Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,0,2]=MAE(f1Unet[i](dT[:,0]), dT[:,1])

        res['res'][0,i,1,0]=RMSE(f2Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,1,1]=MAPELoss(f2Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,1,2]=MAE(f2Unet[i](dT[:,0]), dT[:,1])

        res['res'][0,i,2,0]=RMSE(f3Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,2,1]=MAPELoss(f3Unet[i](dT[:,0]), dT[:,1])
        res['res'][0,i,2,2]=MAE(f3Unet[i](dT[:,0]), dT[:,1])
        if method =='ALL':
            # #KNN
            # x1P,_ = testKNN(data1, None, None, dT)
            # x2P,_ = testKNN(data2, None, None, dT)
            # x3P,_ = testKNN(data3, None, None, dT)
            # res['res'][1,i,0,0]=RMSE(x1P, dT[:,1])
            # res['res'][1,i,0,1]=MAPELoss(x1P, dT[:,1])
            # res['res'][1,i,0,2]=MAE(x1P, dT[:,1])
            #
            # res['res'][1,i,1,0]=RMSE(x2P, dT[:,1])
            # res['res'][1,i,1,1]=MAPELoss(x2P, dT[:,1])
            # res['res'][1,i,1,2]=MAE(x2P, dT[:,1])
            #
            # res['res'][1,i,2,0]=RMSE(x3P, dT[:,1])
            # res['res'][1,i,2,1]=MAPELoss(x3P, dT[:,1])
            # res['res'][1,i,2,2]=MAE(x3P, dT[:,1])
            #SNN
            x1P, _ = testSNN2(data1, None, None, dT,hidden=(50,50))
            x2P, _ = testSNN2(data2, None, None, dT,hidden=(50,50))
            x3P, _ = testSNN2(data3, None, None, dT,hidden=(50,50))
            res['res'][2, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][2, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][2, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][2, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][2, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][2, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][2, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][2, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][2, i, 2, 2] = MAE(x3P, dT[:, 1])
            #DE
            x1P, _ = testCurveModel(dT,k,1,'DE')
            x2P, _ = testCurveModel(dT,k,2,'DE')
            x3P, _ = testCurveModel(dT,k,3,'DE')
            res['res'][3, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][3, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][3, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][3, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][3, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][3, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][3, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][3, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][3, i, 2, 2] = MAE(x3P, dT[:, 1])

            # ADE
            x1P, _ = testCurveModel(dT, k, 1, 'ADE')
            x2P, _ = testCurveModel(dT, k, 2, 'ADE')
            x3P, _ = testCurveModel(dT, k, 3, 'ADE')
            res['res'][4, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][4, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][4, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][4, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][4, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][4, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][4, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][4, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][4, i, 2, 2] = MAE(x3P, dT[:, 1])

            # # PLF4
            x1P, _ = testCurveModel(dT, k, 1, 'PLF4')
            x2P, _ = testCurveModel(dT, k, 2, 'PLF4')
            x3P, _ = testCurveModel(dT, k, 3, 'PLF4')
            res['res'][5, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][5, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][5, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][5, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][5, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][5, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][5, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][5, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][5, i, 2, 2] = MAE(x3P, dT[:, 1])

            # # PLF5
            x1P, _ = testCurveModel(dT, k, 1, 'PLF5')
            x2P, _ = testCurveModel(dT, k, 2, 'PLF5')
            x3P, _ = testCurveModel(dT, k, 3, 'PLF5')
            res['res'][6, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][6, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][6, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][6, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][6, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][6, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][6, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][6, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][6, i, 2, 2] = MAE(x3P, dT[:, 1])

            #Spline
            time_start = time.time()
            x1P, _ = testSpline(data1, None, None, dT)
            x2P, _ = testSpline(data2, None, None, dT)
            x3P, _ = testSpline(data3, None, None, dT)

            res['res'][7, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][7, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][7, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][7, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][7, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][7, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][7, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][7, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][7, i, 2, 2] = MAE(x3P, dT[:, 1])
            time_end = time.time()
            timeCost += time_end - time_start
        elif method=='SRAndUnet':
            x1P, _ = testSpline(data1, None, None, dT)
            x2P, _ = testSpline(data2, None, None, dT)
            x3P, _ = testSpline(data3, None, None, dT)

            res['res'][2, i, 0, 0] = RMSE(x1P, dT[:, 1])
            res['res'][2, i, 0, 1] = MAPELoss(x1P, dT[:, 1])
            res['res'][2, i, 0, 2] = MAE(x1P, dT[:, 1])

            res['res'][2, i, 1, 0] = RMSE(x2P, dT[:, 1])
            res['res'][2, i, 1, 1] = MAPELoss(x2P, dT[:, 1])
            res['res'][2, i, 1, 2] = MAE(x2P, dT[:, 1])

            res['res'][2, i, 2, 0] = RMSE(x3P, dT[:, 1])
            res['res'][2, i, 2, 1] = MAPELoss(x3P, dT[:, 1])
            res['res'][2, i, 2, 2] = MAE(x3P, dT[:, 1])
        # print('SNN time cost', timeCost / (i + 1), 's')
    # if cutdownX:
    #     with open('resCTotal0-8'+str(xPer), 'wb') as f:
    #         pickle.dump(res, f)
    # else:
    #     with open('resTotal0-8T', 'wb') as f:
    #         pickle.dump(res, f)
    return res
if __name__ == '__main__':
    # seq = ["ZMS4WT7", "LN43", "H1-10F", "ZMS1WT6", "ZMS1WT9",
    #        "JF-64", "LN62", "H1-06F", "H1-27F", "A1-05", "H1-13F",
    #        "LN51", "A1-14", "A1-10", "JF-50", "A1-11", "H1-03F", "ZMS1WT7", "H1-02F",
    #        "JF-78", "H1-16F", "ZMS4WT6", "H1-05F", "A1-07", "H1-29F", "A1-08", "ZMS4WT2", "ZMS4WT4",
    #        "A1-13", "H1-19F", "H1-07F", "H1-26F", "H1-08F", "A1-12", "H1-28F", "A2-25", "A2-30", "A1-09",
    #        "H1-09F", "ZMS4WT9", "LN35", "A2-23", "LN40", "ZMS4WT1", "H1-01F", "H1-31F", "ZMS1WT1", "A1-02", "LN41",
    #        "H1-17F", "ZMS1WT8", "LN42", "ZMS4WT3", "LN63", "H1-14F", "ZMS4WT8", "H1-32F", "H1-12F", "H1-11F", "LN54",
    #        "H1-04F", "H1-20F", "H1-24F", "ZMS1WT4", "A2-21", "ZMS1WT2", "ZMS1WT3", "H1-18F", "H1-15F", "JF-33", "A2-22",
    #        "A2-19", "H1-23F", "LN53", "H1-21F", "H1-22F", "A2-33"]



    # testOnce(cutdownX = False,xPer = 0.85 * 100)
    # testOnce(cutdownX = True,xPer = 0.85 * 100,dataType='Ndefault')
    # testOnce(cutdownX = True,xPer = 0.875 * 100,dataType='Ndefault')
    # testOnce(cutdownX = True,xPer = 0.90 * 100,dataType='Ndefault')
    # testOnce(cutdownX = True,xPer = 0.925 * 100,dataType='Ndefault')
    # testOnce(cutdownX = True,xPer = 0.95* 100,dataType='Ndefault')
    # testOnce(cutdownX = True,xPer = 0.975* 100,dataType='Ndefault')
    # res['res'][6, i, 2, 2] = MAE(x3P, dT[:, 1])
    # res1=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='SRAndUnet',domainCorrlation=False)
    # min1 = -res1['res'][2, :, 2, 0] +res1['res'][0, :, 2, 0]
    # for i in range(9):
    #     print(res1['seq'][np.argsort(min1)[-i - 1]])
    #     print(min1[np.argsort(min1)[-i - 1]])
    # print(np.mean(res1['res'][0, :, :, :], axis=0))
    # print(np.mean(res1['res'][2, :, :, :], axis=0))
    # res1=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='Unet',domainCorrlation=False)
    # print(np.mean(res1['res'][0, :, :, :],axis=0))
    res2=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='Unet',domainCorrlation=False,nP=100)
    print(np.mean(res2['res'][0, :, :, :], axis=0))
    res2=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='Unet',domainCorrlation=True,nP=100)
    print(np.mean(res2['res'][0, :, :, :],axis=0))
    # min1 = np.sum(np.abs((res1['res'] - res2['res'])[0, :, :, 0]),axis=1)
    # min2= np.abs((res1['res'] - res2['res'])[0, :, :, 0])
    # print(np.sort(min1,axis=0))
    # print(np.argsort(min1))

    # for i in range(9):
    #     print(res2['seq'][np.argsort(min1)[-i-1]])
    #     print(np.sort(min1,axis=0)[np.argsort(min1)[-i-1]])
    # print(np.mean(res2['res'][0, np.argsort(min1)[:-9], :, :], axis=0))
    # print(np.mean(res1['res'][0, np.argsort(min1)[:-9], :, :], axis=0))
    #
    # res1=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='ALL',domainCorrlation=True)
    # for i in range(8):
    #     print(np.mean(res1['res'][i, :, :, :], axis=0))

    # res2=testOnce(cutdownX=False,minmax=True,dataType='Non', xPer=0.85 * 100,method='Unet',domainCorrlation=True)
    # print(np.mean(res2['res'][0, :, :, :],axis=0))
#result100np
# [[7.06252685e-02 1.17437397e+07 4.71078865e-02]
#  [7.06632718e-02 1.16426846e+07 4.70964957e-02]
#  [6.88971074e-02 1.29987102e+07 4.60240889e-02]]


# [[6.96276487e-02 3.38747196e+06 4.68852765e-02]
#  [6.96803029e-02 3.40144090e+06 4.68451755e-02]
#  [6.89301723e-02 3.02658089e+06 4.61663115e-02]]
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]
# [[7.77352213e-02 4.28584765e+07 5.49238696e-02]
#  [7.35506332e-02 5.58762788e+07 5.10239632e-02]
#  [6.83779084e-02 2.92717508e+07 4.62634235e-02]]
# [[8.42264268e-02 9.77528018e+05 6.13166070e-02]
#  [8.19378128e-02 2.27283989e+06 6.02426737e-02]
#  [7.94562780e-02 2.04049624e+06 5.89380478e-02]]
# [[9.63588701e-02 8.40710255e+07 7.25806697e-02]
#  [9.00010486e-02 4.29237041e+07 6.60506708e-02]
#  [9.13866318e-02 3.64458914e+07 6.75515405e-02]]
# [[7.97532421e-02 6.18621890e+07 5.72814575e-02]
#  [7.56678791e-02 5.15537499e+07 5.42113250e-02]
#  [7.10185487e-02 5.82922282e+07 5.08724990e-02]]
# [[7.95893884e-02 8.71050049e+07 5.68931710e-02]
#  [7.64042584e-02 8.10206132e+07 5.41936548e-02]
#  [7.21370497e-02 8.77580390e+07 5.04852359e-02]]
# [[7.61024785e-02 3.80427297e+06 5.03678451e-02]
#  [7.31877251e-02 4.23740234e+06 4.71833578e-02]
#  [6.86083103e-02 4.25079413e+06 4.39870063e-02]]


























