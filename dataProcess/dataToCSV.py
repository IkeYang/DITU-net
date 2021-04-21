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
warnings.filterwarnings("ignore")
from drawCurve.utlize import DE,ADE,PLF4II

path = r"D:\YANG Luoxiao\Data\WPC\newRes"
path = r"D:\YANG Luoxiao\Data\WPC\newRes"
with open(path, "rb") as f:
    newRes = pickle.load(f)
data = newRes["data"]
thresRes = newRes["thres"]

seq=["ZMS4WT7", "LN43", "H1-10F", "ZMS1WT6", "ZMS1WT9",
     "JF-64", "LN62", "H1-06F", "H1-27F", "A1-05", "H1-13F",
     "LN51", "A1-14", "A1-10", "JF-50", "A1-11", "H1-03F", "ZMS1WT7", "H1-02F",
     "JF-78", "H1-16F", "ZMS4WT6", "H1-05F", "A1-07", "H1-29F", "A1-08", "ZMS4WT2", "ZMS4WT4",
     "A1-13", "H1-19F", "H1-07F", "H1-26F", "H1-08F", "A1-12", "H1-28F", "A2-25", "A2-30", "A1-09",
     "H1-09F", "ZMS4WT9", "LN35", "A2-23", "LN40", "ZMS4WT1", "H1-01F", "H1-31F", "ZMS1WT1", "A1-02", "LN41",
     "H1-17F", "ZMS1WT8", "LN42", "ZMS4WT3", "LN63", "H1-14F", "ZMS4WT8", "H1-32F", "H1-12F", "H1-11F", "LN54",
     "H1-04F", "H1-20F", "H1-24F", "ZMS1WT4", "A2-21", "ZMS1WT2", "ZMS1WT3", "H1-18F", "H1-15F", "JF-33", "A2-22",
     "A2-19", "H1-23F", "LN53", "H1-21F", "H1-22F", "A2-33"]
seq=['H1-30F','H1-25F', 'A1-02', 'LN49']
path=r'D:\YANG Luoxiao\Data\WPC\CSV'
for i,k in enumerate(seq):
    data1, data2, data3, dT = fliterData(data[k], thresRes[k], trainTestSPlit=True, minmax=False)
    np.savetxt(path+r'\\'+k+'1.txt',data1)
    np.savetxt(path+r'\\'+k+'2.txt',data2)
    np.savetxt(path+r'\\'+k+'3.txt',data3)
# from baseline.testCurveModel import testCurveModel
# name='H1-22F'
# j=3
# a=np.loadtxt(path+r'\\'+name+'%s.txt'%(str(j)))
# b=np.loadtxt(path+r'\\'+name+'%sDEres.txt'%(str(j)))
# c=np.loadtxt(path+r'\\'+name+'%sADEres.txt'%(str(j)))
# d=np.loadtxt(path+r'\\'+name+'%sPLF4res.txt'%(str(j)))
# plt.plot(a[:,0],a[:,1],'*')
# plt.plot(a[:,0],testCurveModel(a,name,j,'DE')[0],'*')
# plt.plot(a[:,0],testCurveModel(a,name,j,'ADE')[0],'*')
# plt.plot(a[:,0],testCurveModel(a,name,j,'PLF4')[0],'*',color='black')
# plt.show()










