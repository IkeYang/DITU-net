
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
from scipy import interpolate
from Unet.openImageNp import imageToLine
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





name='jf78N'
path=r'C:\Users\yang\OneDrive\文档\data'
path = r'D:\YANG Luoxiao\Data\gearbox'
filename=name+r'.xlsx'
path=path+ r'\\' + filename

variable=['wind_speed','grid_power']
dataF = pd.read_excel(path)
data=pd.DataFrame()
for i in variable:
    data[i]=dataF[i]
data=data.values

data1,data2,data3,data4=cleanData(data)

x=np.arange(0,1,0.01)

C=10
e=0.0001
paraModel=DEModel()
snnModel=SNNModel()
f,axs=plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        axs[i,j].set_xlim((-0.2, 1.2))
        axs[i,j].set_ylim((-0.2, 1.2))



dataUsed=data1
kind='linear'

device = torch.device('cpu')
model=trainParameterModel(paraModel,dataUsed,device=device)
xt=torch.from_numpy(x).float().to(device)
y=model(xt).detach().numpy()
axs[0,0].plot(dataUsed[:,0],dataUsed[:,1],'*')
axs[0,0].plot(x,y,label='DE')
knn=trainKNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1].reshape((-1,1)),n=30)
y=knn.predict(x.reshape((-1,1)))
axs[0,0].plot(x,y,label='KNN')
svr=trainSVM(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1],C=C,epsilon=e)
y=svr.predict(x.reshape((-1,1)))
axs[0,0].plot(x,y,label='SVR')
# snn=trainSNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1])
# y=snn.predict(x.reshape((-1,1)))
model=trainParameterModel(snnModel,dataUsed,device=device)
xt=torch.from_numpy(x).float().to(device)
y=model(xt.reshape((-1,1))).detach().numpy()
axs[0,0].plot(x,y,label='snn')


dataUsed[:,0]+=np.abs(np.random.randn(len(dataUsed[:,0]))*1e-6)
if len(np.where(dataUsed[:,0]==0)[0])==0:
    dataUsed[0,0]=0
    dataUsed[0,1]=0
if len(np.where(dataUsed[:,0]==1)[0])==0:
    dataUsed[-1,0]=1
    dataUsed[-1,1]=1
datainp=np.sort(dataUsed,axis=0)

f=interpolate.interp1d(datainp[:,0],datainp[:,1],kind=kind)
y_bspline=f(x)
axs[0,0].plot(x,y_bspline,label='spline k=%s'%(kind),lw=1.5)

dataName=name+'data1res'
img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(dataName)
with open(img_nameX, 'rb') as f:
    xim=pickle.load(f)
xi,yi,_=imageToLine(xim)

axs[0,0].plot(xi,yi,label='Unet',lw=1.5)



axs[0,0].legend()



dataUsed=data2

model=trainParameterModel(paraModel,dataUsed,device=device)
xt=torch.from_numpy(x).float().to(device)
y=model(xt).detach().numpy()
axs[0,1].plot(dataUsed[:,0],dataUsed[:,1],'*')
axs[0,1].plot(x,y,label='DE')
knn=trainKNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1].reshape((-1,1)),n=30)
y=knn.predict(x.reshape((-1,1)))
axs[0,1].plot(x,y,label='KNN')
svr=trainSVM(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1],C=C,epsilon=e)
y=svr.predict(x.reshape((-1,1)))
axs[0,1].plot(x,y,label='SVR')
snn=trainSNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1])
y=snn.predict(x.reshape((-1,1)))
axs[0,1].plot(x,y,label='snn')

dataUsed[:,0]+=np.abs(np.random.randn(len(dataUsed[:,0]))*1e-6)
if len(np.where(dataUsed[:,0]==0)[0])==0:
    dataUsed[0,0]=0
    dataUsed[0,1]=0
if len(np.where(dataUsed[:,0]==1)[0])==0:
    dataUsed[-1,0]=1
    dataUsed[-1,1]=1
datainp=np.sort(dataUsed,axis=0)
f=interpolate.interp1d(datainp[:,0],datainp[:,1],kind=kind)
y_bspline=f(x)
axs[0,1].plot(x,y_bspline,label='spline k=%s'%(kind))
dataName=name+'data2res'
img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(dataName)
with open(img_nameX, 'rb') as f:
    xim=pickle.load(f)
xi,yi,_=imageToLine(xim)
axs[0,1].plot(xi,yi,label='Unet')
axs[0,1].legend()



dataUsed=data3

model=trainParameterModel(paraModel,dataUsed,device=device)
xt=torch.from_numpy(x).float().to(device)
y=model(xt).detach().numpy()
axs[1,0].plot(dataUsed[:,0],dataUsed[:,1],'*')
axs[1,0].plot(x,y,label='DE')
knn=trainKNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1].reshape((-1,1)),n=30)
y=knn.predict(x.reshape((-1,1)))
axs[1,0].plot(x,y,label='KNN')
svr=trainSVM(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1],C=C,epsilon=e)
y=svr.predict(x.reshape((-1,1)))
axs[1,0].plot(x,y,label='SVR')
snn=trainSNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1])
y=snn.predict(x.reshape((-1,1)))
axs[1,0].plot(x,y,label='snn')

dataUsed[:,0]+=np.abs(np.random.randn(len(dataUsed[:,0]))*1e-6)
if len(np.where(dataUsed[:,0]==0)[0])==0:
    dataUsed[0,0]=0
    dataUsed[0,1]=0
if len(np.where(dataUsed[:,0]==1)[0])==0:
    dataUsed[-1,0]=1
    dataUsed[-1,1]=1
datainp=np.sort(dataUsed,axis=0)
f=interpolate.interp1d(datainp[:,0],datainp[:,1],kind=kind)
y_bspline=f(x)
axs[1,0].plot(x,y_bspline,label='spline k=%s'%(kind))

dataName=name+'data3res'
img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(dataName)
with open(img_nameX, 'rb') as f:
    xim=pickle.load(f)
xi,yi,_=imageToLine(xim)
axs[1,0].plot(xi,yi,label='Unet')

axs[1,0].legend()
dataUsed=data4

model=trainParameterModel(paraModel,dataUsed,device=device)
xt=torch.from_numpy(x).float().to(device)
y=model(xt).detach().numpy()
axs[1,1].plot(dataUsed[:,0],dataUsed[:,1],'*')
axs[1,1].plot(x,y,label='DE')
knn=trainKNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1].reshape((-1,1)),n=30)
y=knn.predict(x.reshape((-1,1)))
axs[1,1].plot(x,y,label='KNN')
svr=trainSVM(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1],C=C,epsilon=e)
y=svr.predict(x.reshape((-1,1)))
axs[1,1].plot(x,y,label='SVR')
snn=trainSNN(dataUsed[:,0].reshape((-1,1)),dataUsed[:,1])
y=snn.predict(x.reshape((-1,1)))
axs[1,1].plot(x,y,label='snn')


dataUsed[:,0]+=np.abs(np.random.randn(len(dataUsed[:,0]))*1e-6)
if len(np.where(dataUsed[:,0]==0)[0])==0:
    dataUsed[0,0]=0
    dataUsed[0,1]=0
if len(np.where(dataUsed[:,0]==1)[0])==0:
    dataUsed[-1,0]=1
    dataUsed[-1,1]=1
datainp=np.sort(dataUsed,axis=0)
f=interpolate.interp1d(datainp[:,0],datainp[:,1],kind=kind)
y_bspline=f(x)
axs[1,1].plot(x,y_bspline,label='spline k=%s'%(kind))

dataName=name+'data4res'
img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(dataName)
with open(img_nameX, 'rb') as f:
    xim=pickle.load(f)
xi,yi,_=imageToLine(xim)
axs[1,1].plot(xi,yi,label='Unet')
axs[1,1].legend()





plt.show()


















