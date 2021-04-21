import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import pickle
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"C:\MyPhDCde\我的坚果云\windPowerCurveModeling")
from baseline.model import trainParameterModel
from baseline.model import DEModel
from baseline.model import SNNModel
from baseline.model import trainKNN
from baseline.model import trainSNN
from baseline.model import trainSVM
from baseline.model import trainSpline
from scipy import interpolate
device = torch.device('cpu')
from sklearn.metrics import mean_squared_error
from dataProcess.clearWithCoord import fliterData
#
# def testDE(data1,data2,data3,dataT):
#     paraModel = DEModel()
#
#     model = trainParameterModel(paraModel, data1, device=device)
#     xt = torch.from_numpy(dataT[:,0]).float().to(device).view(-1,1)
#     x1P = model(xt).detach().numpy().flatten()
#     if data2 is None:
#         return x1P
#     model = trainParameterModel(paraModel, data2, device=device)
#     xt = torch.from_numpy(dataT[:,0]).float().to(device).view(-1,1)
#     x2P = model(xt).detach().numpy().flatten()
#
#
#     model = trainParameterModel(paraModel, data3, device=device)
#     xt = torch.from_numpy(dataT[:,0]).float().to(device).view(-1,1)
#     x3P = model(xt).detach().numpy().flatten()
#
#
#     return np.sqrt(mean_squared_error(x1P, dataT[:,1])),np.sqrt(mean_squared_error(x2P, dataT[:,1])),np.sqrt(mean_squared_error(x3P, dataT[:,1]))
#
#
# def testSVR(data1, data2, data3, dataT,C=10,e=0.0001):
#
#
#     svr = trainSVM(data1[:, 0].reshape((-1, 1)), data1[:, 1], C=C, epsilon=e)
#     x1P = svr.predict(dataT[:,0].reshape((-1, 1)))
#     if data2 is None:
#         return x1P
#     svr = trainSVM(data2[:, 0].reshape((-1, 1)), data2[:, 1], C=C, epsilon=e)
#     x2P = svr.predict(dataT[:,0].reshape((-1, 1)))
#
#     svr = trainSVM(data3[:, 0].reshape((-1, 1)), data3[:, 1], C=C, epsilon=e)
#     x3P = svr.predict(dataT[:,0].reshape((-1, 1)))
#
#     return np.sqrt(mean_squared_error(x1P, dataT[:, 1])), np.sqrt(mean_squared_error(x2P, dataT[:, 1])), np.sqrt(
#         mean_squared_error(x3P, dataT[:, 1]))


def testSNN(data1, data2, data3, dataT):
    paraModel = SNNModel()

    model = trainParameterModel(paraModel, data1, device=device)
    xt = torch.from_numpy(dataT[:,0]).float().to(device).view(-1,1)
    x1P = model(xt).detach().numpy().flatten()
    if data2 is None:
        return x1P,model
    model = trainParameterModel(paraModel, data2, device=device)
    xt = torch.from_numpy(dataT[:,0]).float().to(device).view(-1,1)
    x2P = model(xt).detach().numpy().flatten()

    model = trainParameterModel(paraModel, data3, device=device)
    xt = torch.from_numpy(dataT[:,1]).float().to(device).view(-1,1)
    x3P = model(xt).detach().numpy().flatten()

    return np.sqrt(mean_squared_error(x1P, dataT[:, 1])), np.sqrt(mean_squared_error(x2P, dataT[:, 1])), np.sqrt(
        mean_squared_error(x3P, dataT[:, 1]))
def testSNN2(data1, data2, data3, dataT,hidden=(5000,)):
    knn = trainSNN(data1[:, 0].reshape((-1, 1)), data1[:, 1].reshape((-1, 1)),hidden_layer_sizes=hidden)
    x1P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()
    if data2 is None:
        return x1P,knn
    knn = trainSNN(data2[:, 0].reshape((-1, 1)), data2[:, 1].reshape((-1, 1)))
    x2P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()

    knn = trainSNN(data3[:, 0].reshape((-1, 1)), data3[:, 1].reshape((-1, 1)))
    x3P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()

    return np.sqrt(mean_squared_error(x1P, dataT[:, 1])), np.sqrt(mean_squared_error(x2P, dataT[:, 1])), np.sqrt(
        mean_squared_error(x3P, dataT[:, 1]))

def testKNN(data1, data2, data3, dataT, n=30):
    knn = trainKNN(data1[:, 0].reshape((-1, 1)), data1[:, 1].reshape((-1, 1)), n=n)
    x1P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()
    if data2 is None:
        return x1P,knn
    knn = trainKNN(data2[:, 0].reshape((-1, 1)), data2[:, 1].reshape((-1, 1)), n=n)
    x2P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()

    knn = trainKNN(data3[:, 0].reshape((-1, 1)), data3[:, 1].reshape((-1, 1)), n=n)
    x3P = knn.predict(dataT[:,0].reshape((-1, 1))).flatten()

    return np.sqrt(mean_squared_error(x1P, dataT[:, 1])), np.sqrt(mean_squared_error(x2P, dataT[:, 1])), np.sqrt(
        mean_squared_error(x3P, dataT[:, 1]))


def testSpline(data1, data2, data3, dataT, kind='linear',cut=False):

    f=trainSpline(data1, savename=None, kind=kind,cut=cut)
    x1P = f(dataT[:, 0])
    if data2 is None:
        return x1P,f
    f = trainSpline(data2, savename=None, kind=kind,cut=cut)
    x2P = f(dataT[:, 0])

    f = trainSpline(data3, savename=None, kind=kind,cut=cut)
    x3P = f(dataT[:, 0])

    return np.sqrt(mean_squared_error(x1P, dataT[:, 1])), np.sqrt(mean_squared_error(x2P, dataT[:, 1])), np.sqrt(
        mean_squared_error(x3P, dataT[:, 1]))


if __name__ == '__main__':

    path = r'D:\YANG Luoxiao\Data\WPC\WPC'
    path = r'D:\YANG Luoxiao\Data\WPC\ZMD'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'rb') as f:
        thresRes = pickle.load(f)

    for epoch in range(1, 10):
        i = 0
        r1M = 0
        r2M = 0
        r3M = 0
        print(epoch)
        for (k, v) in thresRes.items():

            i += 1
            data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True,minmax=False)
            r1, r2, r3 = testSpline(data1, data2, data3, dT)
            print(k,r1,r2,r3)
            r1M += r1
            r2M += r2
            r3M += r3
        print(r1M / i, r2M / i, r3M / i)
