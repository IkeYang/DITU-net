import pickle
import numpy as np
import matplotlib.pyplot as plt
amL= ['ZMS1WT7', 'A2-30', 'ZMS1WT2', 'A1-14', 'A2-33', 'A1-12', 'A1-09', 'JF-50', 'A2-22', 'ZMS1WT3', 'ZMS1WT8', 'LN35', 'ZMS1WT1', 'A1-13', 'A1-10', 'A1-11', 'A2-19', 'JF-64', 'LN41', 'ZMS1WT6', 'A1-08', 'A2-21', 'A1-02', 'LN42', 'LN40', 'LN49', 'A1-07', 'LN51', 'LN53', 'LN43', 'A1-05', 'JF-78', 'ZMS1WT4', 'A2-23', 'LN62', 'A2-25', 'LN54', 'LN63', 'ZMS1WT9', 'JF-33']


def returnIndex(seq,L1):
    bL=[]
    mL=[]
    for i,k in enumerate(seq):
        if k in L1:
            mL.append(i)
        else:
            bL.append(i)
    return bL,mL






with open('resTotal0-8T', 'rb') as f:
# with open('resTotal0-8', 'rb') as f:
# with open('resCTotal0-885.0', 'rb') as f:
# with open('resTotal0-9', 'rb') as f:
    res=pickle.load(f)
# print(res)
# print(res['seq'])
# print(res['rate'])
print(res['methodOrder'])
av=np.mean(res['res'],axis=(1))
print(av[:,:,0])
print(av[:,:,1])
print(av[:,:,2])

bL,mL=returnIndex(res['seq'],amL)

print('bL************************')
av=np.mean(res['res'][:,bL,:,:],axis=(1))
print(av[:,:,0])
print(av[:,:,1])
print(av[:,:,2])


print('mL************************')
av=np.mean(res['res'][:,mL,:,:],axis=(1))
print(av[:,:,0])
print(av[:,:,1])
print(av[:,:,2])