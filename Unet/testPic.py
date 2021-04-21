import os
import sys
# sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"C:\MyPhDCde\我的坚果云\windPowerCurveModeling")
import torch

import pandas as pd
from skimage import io, transform
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from Unet.model import UNet,transform_invert
from dataProcess.sparsePictEnhence import checkAndEnhancYregion
from dataProcess.clearWithCoord import fliterData
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sympy
from sklearn.neighbors import KNeighborsRegressor
# x = x[~np.isnan(x)]
from functools import wraps
import time

# def getRealRoot(f):
#     x = sympy.symbols('x')
#     a=[]
#     aa = sympy.solve(x ** 3 + 2 * x + 3, x)
#     for i in aa:
#         try:
#             a.append(np.array(i).astype(np.float64))
#         except:
#             pass
#     return np.array(a)
def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''

    @wraps(function)
    def function_timer(*args, **kwargs):
        print
        ('[Function: {name} start...]'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print
        ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def testPict(data,name,path=r'D:\YANG Luoxiao\Data\gearbox',orginal=True,scaler=None,plot=True,epoch=1,ms=None):
    # name='jf78N'
    #     filename=name+r'.xlsx'
    #     path=path+ r'\\' + filename
    #
    #     variable=['wind_speed','grid_power']
    #     dataF = pd.read_excel(path)
    #     data=pd.DataFrame()
    #     for i in variable:
    #         data[i]=dataF[i]
    #     data=data.values
    if plot:
        print(len(data))
    rate=1
    data=data[:int(len(data)*rate),:]
    if plot:
        print(len(data))

    if orginal:
        min_max_scaler = MinMaxScaler()
        data=min_max_scaler.fit_transform(data)
        scaler=min_max_scaler
    else:
        data = scaler.transform(data)
    f, ax = plt.subplots(1, 1,figsize=(2.56, 2.56))
    ind=np.where(data[:,1]>0.6)
    # ax.plot(data[ind,0]+np.random.randn(len(ind))*0.01,data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0]+np.random.randn(len(ind))*0.01,data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0],data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0],data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    if ms is None:
        if len(data)>10000:
            # ax.plot(data[:,0],data[:,1],'.',color='black',markersize=1)
            ax.plot(data[:,0],data[:,1],'.',color='black',markersize=ms)
        else:
            ax.plot(data[:,0],data[:,1],'.',color='black')
    else:
        ax.plot(data[:, 0], data[:, 1], '.', color='black', markersize=ms)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    savePath=r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
    name1=savePath+r'\\'+name
    f.savefig(name1+'.jpg', dpi=100)
    plt.close(f)
    # plt.show()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirP=r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
    img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly\%s.jpg"%(name)
    imageX = Image.open(img_nameX).convert('RGB')

    transf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    imageX = transf(imageX).view(1,3,256,256).to(device)




    outf=r"D:\YANG Luoxiao\Model\WPC\DE\Unet"
    unet = UNet(in_channels=3, out_channels=3)
    unet.to(device)
    unet.eval()
    setLen=4000
    # checkpoint = torch.load('%s/NoNormUnet800DE.pth' % (outf))
    checkpoint = torch.load('%s/Unet800DE.pth' % (outf))
    checkpoint = torch.load('%s/UnetS2000DE.pth' % (outf))
    # checkpoint = torch.load('%s/UnetS5000DEAndADE.pth' % (outf))
    # checkpoint = torch.load('%s/UnetS5000DEAndADELast.pth' % (outf))
    checkpoint = torch.load('%s/UnetS%dDEAndADEepoch%d.pth' % (outf,setLen,epoch))
    checkpoint = torch.load('%s/UnetS%dDEAndADECutepoch%d.pth' % (outf,setLen,epoch))
    if epoch==11:
        checkpoint = torch.load('%s/UnetS2000DE.pth' % (outf))
    unet.load_state_dict(checkpoint['model'])




    savePath=r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes"
    ypred=unet(imageX)
    x=imageX.detach().cpu().squeeze()
    ypred=ypred.detach().cpu().squeeze()
    yR=transform_invert(ypred,transf)
    with open(savePath+r'\\'+name+'resNp','wb') as f:
        pickle.dump(yR,f)
    xy= transform_invert(x+ypred, transf)
    plt.imshow(xy)


    name1=savePath+r'\\'+name
    plt.savefig(name1+'res.jpg', dpi=100)
    if plot:
        plt.show()

    y= transform_invert(ypred, transf)
    plt.imshow(y)
    if plot:
        plt.show()
    return yR,scaler

# testPict(name='jf78N',path=r'D:\YANG Luoxiao\Data\gearbox')
# testPict(name='jf50N',path=r'D:\YANG Luoxiao\Data\gearbox')
# testPict(name='jf33N',path=r'D:\YANG Luoxiao\Data\gearbox')
# testPict(name='jf64',path=r'D:\YANG Luoxiao\Data\gearbox')


def plotPict(data,name,nP=200):
    ms=np.sqrt(nP*36/data.shape[0])#200 is good
    f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
    ind = np.where(data[:, 1] > 0.6)
    # ax.plot(data[ind,0]+np.random.randn(len(ind))*0.01,data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0]+np.random.randn(len(ind))*0.01,data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0],data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    # ax.plot(data[ind,0],data[ind,1]+np.random.randn(len(ind))*0.01,'.',color='black')
    if ms is None:
        if len(data) > 10000:
            ax.plot(data[:, 0], data[:, 1], '.', color='black', markersize=ms)
        else:
            ax.plot(data[:, 0], data[:, 1], '.', color='black')
    else:
        ax.plot(data[:, 0], data[:, 1], '.', color='black', markersize=ms)
        # ax.plot(data[:, 0], data[:, 1], '.', color='black', markersize=6)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    savePath = r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
    name1 = savePath + r'\\' + name
    f.savefig(name1 + '.jpg', dpi=100)
    plt.close(f)
    img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly\%s.jpg"%(name)
    imageX = Image.open(img_nameX).convert('RGB')

    transf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    imageX = transf(imageX).view(1,3,256,256)
    return imageX

def dataRate(data1, data2, data3, dT):
    wholeLen=data1.shape[0]
    r1Len=data2.shape[0]
    r2Len=data3.shape[0]
    return r1Len/wholeLen, r2Len/wholeLen


def dataToImageBench(data,thresRes,ms=None,minmax=True,seq=False,rateR=False,
                     enhance=False,dev=10,testOnly=False,cutdownX=False,xPer=0.9,nP=200):
    '''

    :param data: dict k,v
    :return: data1(bs,c,256,256) data2(bs,c,256,256) data3(bs,c,256,256) torch tensor
    '''
    data1Torch=torch.zeros(1,3,256,256)
    data2Torch=torch.zeros(1,3,256,256)
    data3Torch=torch.zeros(1,3,256,256)
    dataTTorch=torch.zeros(1,3,256,256)
    dictSeq=[]
    dtL=[]
    rate=[]
    if testOnly:
        for (k, v) in thresRes.items():
            dictSeq.append(k)

            data1=data[k]
            # data1 = data1[~np.isnan(data1[:, 0]), :]
            # data1 = data1[~np.isnan(data1[:, 1]), :]
            #
            # number = len(np.where(data1[:, 1] > 0.99)[0])
            # # print(number)
            # if number < 10:
            #     data1[np.where(data1[:, 1] > 0.99), :] = 0
            #     data1[:, 1] = data1[:, 1] / np.max(data1[:, 1])

            if cutdownX:

                num=np.percentile(data1[:,0], xPer)
                ind=data1[:,0]<num
                data1=data1[ind,:]



            if enhance:
                data1 = checkAndEnhancYregion(data1, dev)

            data1Torch = torch.cat((data1Torch, plotPict(data=data1, name=k,nP=nP)), dim=0)

        if seq:

            return data1Torch[1:, :, :, :],dictSeq

        return data1Torch[1:, :, :, :]

    for (k,v) in thresRes.items():
        dictSeq.append(k)

        data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True,minmax=minmax,xcutDown=cutdownX,xpercentilep=xPer)
        r1,r2=dataRate(data1, data2, data3, dT)
        dtL.append(dT)
        rate.append((r1,r2))
        if enhance:
            dataK=checkAndEnhancYregion(data1,dev)
            data1, data2, data3, dT = fliterData(dataK, thresRes[k], trainTestSPlit=True,minmax=minmax,xcutDown=cutdownX,xpercentilep=xPer)
        # min_max_scaler = MinMaxScaler()
        # data1 = min_max_scaler.fit_transform(data1)
        # data2 = min_max_scaler.transform(data2)
        # data3 = min_max_scaler.transform(data3)
        # dT = min_max_scaler.transform(dT)
        data1Torch=torch.cat((data1Torch,plotPict(data=data1,name=k,nP=nP)),dim=0)
        data2Torch=torch.cat((data2Torch,plotPict(data=data2,name=k,nP=nP)),dim=0)
        data3Torch=torch.cat((data3Torch,plotPict(data=data3,name=k,nP=nP)),dim=0)
        # dataTTorch=torch.cat((dataTTorch,plotPict(data=dT,name=k,ms=ms)),dim=0)
    if seq:
        if rateR:
            return data1Torch[1:, :, :, :], data2Torch[1:, :, :, :], data3Torch[1:, :, :, :], dataTTorch[1:, :, :, :], dtL,dictSeq,rate
        return data1Torch[1:, :, :, :], data2Torch[1:, :, :, :], data3Torch[1:, :, :, :], dataTTorch[1:, :, :, :], dtL,dictSeq

    return data1Torch[1:,:,:,:],data2Torch[1:,:,:,:],data3Torch[1:,:,:,:],dataTTorch[1:,:,:,:],dtL

def benchTest(data,epoch=1,bs=16,name='DEAndADE',setLen = 4000,saveMiddle=False):
    '''

    :param data: bs c 256 256
    :param plot:
    :param epoch:
    :param ms:
    :return: yR bs,256,256
    '''

    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=data.float().to(device)
    outf = r"D:\YANG Luoxiao\Model\WPC\DE\Unet"
    unet = UNet(in_channels=3, out_channels=3)
    unet.to(device)
    unet.eval()


    if epoch == 20:
        checkpoint = torch.load('%s/UnetS2000DE.pth' % (outf))
    else:
        # checkpoint = torch.load('%s/NoNormUnet800DE.pth' % (outf))
        checkpoint = torch.load('%s/Unet800DE.pth' % (outf))
        checkpoint = torch.load('%s/UnetS2000DE.pth' % (outf))
        # checkpoint = torch.load('%s/UnetS5000DEAndADE.pth' % (outf))
        # checkpoint = torch.load('%s/UnetS5000DEAndADELast.pth' % (outf))
        checkpoint = torch.load('%s/UnetS%d%sepoch%d.pth' % (outf, setLen, name,epoch))
        # checkpoint = torch.load('%s/UnetS%dDEAndADECutepoch%d.pth' % (outf, setLen, epoch))
    unet.load_state_dict(checkpoint['model'])
    if data.shape[0] % bs ==0:
        iter=int(data.shape[0]/bs)
    else:
        iter = int(data.shape[0] / bs) +1

    if saveMiddle:



        for i in range(iter):
            ypred = unet(data[i * bs:(i + 1) * bs, :, :, :])

            ly = ypred.shape[0]
            ypred = ypred.detach().cpu().squeeze()

            if i == 0:
                yR = transform_invert(ypred, transf).reshape((ly, 256, 256))
                xy= transform_invert(ypred+data[i * bs:(i + 1) * bs, :, :, :].detach().cpu().squeeze(), transf).reshape((ly, 256, 256))
                x= transform_invert(data[i * bs:(i + 1) * bs, :, :, :].detach().cpu().squeeze(), transf).reshape((ly, 256, 256))
            else:
                yR = np.concatenate((yR, transform_invert(ypred, transf).reshape((ly, 256, 256))), axis=0)
                xy = np.concatenate((xy, transform_invert(ypred+data[i * bs:(i + 1) * bs, :, :, :].detach().cpu().squeeze(), transf).reshape((ly, 256, 256))), axis=0)
                x = np.concatenate((x, transform_invert(data[i * bs:(i + 1) * bs, :, :, :].detach().cpu().squeeze(), transf).reshape((ly, 256, 256))), axis=0)
        return yR,(xy,x)

    else:
        for i in range(iter):
            ypred = unet(data[i*bs:(i+1)*bs,:,:,:])

            ly=ypred.shape[0]
            ypred = ypred.detach().cpu().squeeze()


            if i==0:
                yR = transform_invert(ypred, transf).reshape((ly,256,256))
            else:
                yR = np.concatenate((yR,transform_invert(ypred, transf).reshape((ly,256,256))),axis=0)

        return yR
from Unet.openImageNp import imageToLine
from sklearn.metrics import mean_squared_error
import time
def benchImageTest(yr,dtL=None,testOnly=False,p=15,domainCorrlation=True):
    bs=yr.shape[0]
    res=np.zeros([bs,])
    func=[]
    if testOnly:
        for i in range(bs):

            xi, yi, f1 = imageToLine(yr[i, :, :],p=p,domainCorrlation=domainCorrlation)
            func.append(f1)
        return func
    for i in range(bs):
        dt=dtL[i]
        # s=time.time()
        xi, yi, f1 = imageToLine(yr[i,:,:],p=p,domainCorrlation=domainCorrlation)
        # e=time.time()
        # print(e-s)
        func.append(f1)
        xP = f1(dt[:, 0])
        res[i]=np.sqrt(mean_squared_error(xP, dt[:,1]))
    return res,func

def cleanData(data):
    '''

    :param data: len,2(ws,wp)
    :return:data, dataslightly, datahuge1,datahuge2
    '''

    min_max_scaler = MinMaxScaler()
    data=min_max_scaler.fit_transform(data)


    ind=np.where((data[:,0]>0.27 ))
    ind2=np.where((data[:,1]<0.04 ))
    ind=list(set(ind[0]) & set(ind2[0]))
    dataslightly=np.delete(data, ind, 0)
    # data[-ind[0],:]


    ind = np.where(dataslightly[:, 0] > 0.4)
    ind2 = np.where(dataslightly[:, 1] < 0.24)
    ind=list(set(ind[0]) & set(ind2[0]))
    # datahuge1 = dataslightly[-ind[0],:]
    datahuge1 = np.delete(dataslightly, ind, 0)

    ind = np.where((datahuge1[:, 0] > 0.44))
    ind2 = np.where((datahuge1[:, 1] < 0.41))
    ind=list(set(ind[0]) & set(ind2[0]))
    # datahuge2 = datahuge1[-ind[0],:]
    datahuge2 = np.delete(datahuge1, ind, 0)

    ind = np.where((datahuge2[:, 0] > 0.6))
    ind2 = np.where((datahuge2[:, 1] < 0.9))
    ind=list(set(ind[0]) & set(ind2[0]))
    # datahuge2 = datahuge1[-ind[0],:]
    datahuge2 = np.delete(datahuge2, ind, 0)
    return data, dataslightly, datahuge1,datahuge2



def pipLineTestUnet(data,thresRes,epoch=3,ms=3,name='DEAndADE',setLen = 4000,bs=16,enhance=True,dev=10,
                    testOnly=False,dataType='default',saveMiddle=False,cutdownX=False,xPer=0.9,p=12,domainCorrlation=False,nP=200):
    '''

    :param data: input scada data format: type dict {K,v} k name, v the [bs,(ws,wp)] data
    :param thresRes: the cut points of picture the data where x> thresx y<thresy will be eliminated, dict type dict {K,v} k name, v the [number of cut,2]
    :param epoch:which epoch of model default is 3
    :param ms:marker size of plot
    :param name:model name
    :param setLen: training set length
    :param bs: testing bs depends on GPU memory
    :param enhance:if the data is sparse in some space, we will enhance these space
    :param dev: nue number of devision of enhance
    :param testOnly:if test only the data will not be divided
    :param dataType:speed up for model testing
    :param saveMiddle: middle tensor including xy,x
    :return:
    '''
    if thresRes is None:
        thresRes={}
        for k,_ in data.items():
            thresRes[k]=np.zeros([2,2])

    if testOnly:
        data1Torch ,seq =  dataToImageBench(data, thresRes, ms=ms,minmax=False, seq=True,rateR=True,enhance=enhance,
                                            dev=dev,testOnly=testOnly,cutdownX=cutdownX,xPer=xPer,nP=nP)
        if saveMiddle:
            yR,mid1=benchTest(data1Torch, epoch=epoch, bs=bs, name=name, setLen=setLen,saveMiddle=saveMiddle)
            f1 = benchImageTest(yR, testOnly=testOnly,p=p,
                              domainCorrlation=domainCorrlation)
            return seq,yR, f1,mid1
        else:
            yR = benchTest(data1Torch, epoch=epoch, bs=bs, name=name, setLen=setLen,saveMiddle=saveMiddle)
            f1 = benchImageTest(yR,testOnly=testOnly,p=p,
                              domainCorrlation=domainCorrlation)
            return seq, f1
    else:
        path= r'D:\YANG Luoxiao\Data\WPC\dataDefult'
        if dataType =='default':
            if enhance==True and cutdownX==True:
                with open(path + 'CEms%s' % (str(ms)), 'rb') as f:
                    data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate = pickle.load(f)
            if enhance==True and cutdownX==False:
                with open(path + 'Ems%s' % (str(ms)), 'rb') as f:
                    data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate = pickle.load(f)
            else:
                with open(path+'ms%s'%(str(ms)), 'rb') as f:
                    data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate=pickle.load(f)
        else:

            data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate = dataToImageBench(data, thresRes, ms=ms,
                                                                                              minmax=False, seq=True,
                                                                                        rateR=True,enhance=enhance,dev=dev,testOnly=testOnly,
                                                                                              cutdownX=cutdownX,xPer=xPer,nP=nP)

            if enhance == True and cutdownX==True:
                try:
                    with open(path + 'CEms%s' % (str(ms))+str(xPer), 'rb') as f:
                        pass
                except:
                    with open(path + 'CEms%s' % (str(ms))+str(xPer), 'wb') as f:
                        pickle.dump((data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate), f)
            if enhance == True and cutdownX == False:
                try:
                    with open(path + 'Ems%s' % (str(ms)), 'rb') as f:
                        pass
                except:
                    with open(path + 'Ems%s' % (str(ms)), 'wb') as f:
                        pickle.dump((data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate), f)
            else:

                try:
                    with open(path + 'ms%s' % (str(ms)), 'rb') as f:
                        pass
                except:
                    with open(path + 'ms%s' % (str(ms)), 'wb') as f:
                        pickle.dump((data1Torch, data2Torch, data3Torch, dataTTorch, dtL, seq, rate), f)



        rate=np.array(rate)
        if saveMiddle:
            yR,mid1 = benchTest(data1Torch, epoch=epoch, bs=bs, name=name, setLen=setLen, saveMiddle=saveMiddle)
        else:
            yR = benchTest(data1Torch, epoch=epoch, bs=bs,name=name,setLen=setLen,saveMiddle=saveMiddle)
        res1 ,f1= benchImageTest(yR, dtL,p=p,
                              domainCorrlation=domainCorrlation)

        if saveMiddle:
            yR, mid2 = benchTest(data2Torch, epoch=epoch, bs=bs, name=name, setLen=setLen, saveMiddle=saveMiddle)
        else:
            yR = benchTest(data2Torch, epoch=epoch, bs=bs, name=name, setLen=setLen, saveMiddle=saveMiddle)
        res2 ,f2= benchImageTest(yR, dtL,p=p,
                              domainCorrlation=domainCorrlation)

        if saveMiddle:
            yR, mid3 = benchTest(data3Torch, epoch=epoch, bs=bs, name=name, setLen=setLen, saveMiddle=saveMiddle)
        else:
            yR = benchTest(data3Torch, epoch=epoch, bs=bs, name=name, setLen=setLen, saveMiddle=saveMiddle)
        res3,f3 = benchImageTest(yR, dtL,p=p,
                              domainCorrlation=domainCorrlation)
        if saveMiddle:
            return seq, rate, res1, res2, res3, (f1, f2, f3),(mid1,mid2,mid3)
        else:
            return seq,rate,res1,res2,res3,(f1,f2,f3)





if __name__=='__main__':

    # LN35  JF-50 H1-01F A1-01
    name='H1-03F'
    k=name

    # data=data[name]
    # thresRes=thresRes[name]
    data1,data2,data3,dT=fliterData(data[k],thresRes[k],trainTestSPlit=True)
    # data2[data2[:, 1] > 0.85, :] = 0

    _,scaler=testPict(data1,name+'data1',path=r'D:\YANG Luoxiao\Data\gearbox',orginal=True)

    testPict(data2,name+'data2',path=r'D:\YANG Luoxiao\Data\gearbox',orginal=False,scaler=scaler)
    # data3[data3[:, 1] > 0.85, :] = 0
    testPict(data3,name+'data3',path=r'D:\YANG Luoxiao\Data\gearbox',orginal=False,scaler=scaler)
    # testPict(data4,name+'data4',path=r'D:\YANG Luoxiao\Data\gearbox',orginal=False,scaler=scaler)
    import warnings
    warnings.filterwarnings("ignore")

    path=r'D:\YANG Luoxiao\Data\WPC\WPC'
    with open(path,'rb') as f:
        data=pickle.load(f)
    with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
        thresRes=pickle.load(f)

    time_start = time.time()
    data1Torch,data2Torch,data3Torch,dataTTorch,dtL=dataToImageBench(data, thresRes, ms=None)
    time_end = time.time()
    print('time cost dataToImageBench', time_end - time_start, 's')

    time_start = time.time()
    yR=benchTest(data1Torch, epoch=1,bs=16)
    time_end = time.time()
    print('time cost data1Torch', time_end - time_start, 's')

    time_start = time.time()
    res=benchImageTest(yR, dtL)
    time_end = time.time()
    print('time cost benchImageTest', time_end - time_start, 's')

    print(res)





















