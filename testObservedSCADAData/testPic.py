
import sys
sys.path.append(r"..")
import torch
import pickle
from torchvision import transforms, utils
from PIL import Image
from train.model import UNet,transform_invert
from Pixel_mapping_correction  import imageToLine
import numpy as np
import matplotlib.pyplot as plt

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

def plotPict(data,name,nP=200):
    ms=np.sqrt(nP*36/data.shape[0])#200 is good
    f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))

    ax.plot(data[:, 0], data[:, 1], '.', color='black', markersize=ms)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    savePath = r"savePict"
    name1 = savePath + r'\\' + name
    f.savefig(name1 + '.jpg', dpi=100)
    plt.close(f)
    img_nameX = r"%s\%s.jpg"%(savePath,name)
    imageX = Image.open(img_nameX).convert('RGB')

    transf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    imageX = transf(imageX).view(1,3,256,256)
    return imageX




def dataToImageBench(data,thresRes,enhance=False,dev=10,cutdownX=False,xPer=0.9,nP=200):
    '''

    :param data: dict k,v
    :return: data1(bs,c,256,256) data2(bs,c,256,256) data3(bs,c,256,256) torch tensor
    '''
    data1Torch=torch.zeros(1,3,256,256)


    for (k, v) in thresRes.items():

        data1=data[k]
        if cutdownX:
            num=np.percentile(data1[:,0], xPer)
            ind=data1[:,0]<num
            data1=data1[ind,:]
        if enhance:
            data1 = checkAndEnhancYregion(data1, dev)

        data1Torch = torch.cat((data1Torch, plotPict(data=data1, name=k,nP=nP)), dim=0)


    return data1Torch[1:, :, :, :]


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
    outf = r"..\model"
    unet = UNet(in_channels=3, out_channels=3)
    unet.to(device)
    unet.eval()
    checkpoint = torch.load('%s/UnetS%d%sepoch%d.pth' % (outf, setLen, name,epoch))
    unet.load_state_dict(checkpoint['model'])
    if data.shape[0] % bs ==0:
        iter=int(data.shape[0]/bs)
    else:
        iter = int(data.shape[0] / bs) +1

    for i in range(iter):
        ypred = unet(data[i*bs:(i+1)*bs,:,:,:])

        ly=ypred.shape[0]
        ypred = ypred.detach().cpu().squeeze()


        if i==0:
            yR = transform_invert(ypred, transf).reshape((ly,256,256))
        else:
            yR = np.concatenate((yR,transform_invert(ypred, transf).reshape((ly,256,256))),axis=0)

    return yR

from sklearn.metrics import mean_squared_error
import time
def benchImageTest(yr,p=15,domainCorrlation=True):
    bs=yr.shape[0]

    func=[]
    for i in range(bs):

        xi, yi, f1 = imageToLine(yr[i, :, :],p=p,domainCorrlation=domainCorrlation)
        func.append(f1)
    return func






def pipLineTestUnet(data,thresRes=None,epoch=3,name='DEAndADE',setLen = 4000,bs=16,enhance=True,dev=10,saveMiddle=False,cutdownX=False,xPer=0.9,p=12,domainCorrlation=True,nP=200):
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

    data1Torch  =  dataToImageBench(data, thresRes, enhance=enhance,
                                        dev=dev,cutdownX=cutdownX,xPer=xPer,nP=nP)
    yR = benchTest(data1Torch, epoch=epoch, bs=bs, name=name, setLen=setLen,saveMiddle=saveMiddle)
    f1 = benchImageTest(yR,p=p,domainCorrlation=domainCorrlation)
    return  f1





























