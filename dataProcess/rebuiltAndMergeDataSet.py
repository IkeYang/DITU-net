import pickle
import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
sys.path.append(r"C:\MyPhDCde\我的坚果云\windPowerCurveModeling")
import numpy as np
from sklearn.preprocessing import MinMaxScaler


deleteList=['A2-18','A1-01','A1-06']

newRes={}
newRes['data']={}
newRes['thres']={}
path = r'D:\YANG Luoxiao\Data\WPC\WPC'
with open(path, 'rb') as f:
    data = pickle.load(f)

with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    thresRes = pickle.load(f)

for (k,v) in thresRes.items():
    if k in deleteList:
        continue
    else:

        min_max_scaler = MinMaxScaler()
        newRes['data'][k]= min_max_scaler.fit_transform(data[k])
        newRes['thres'][k] = min_max_scaler.transform(thresRes[k])

path = r'D:\YANG Luoxiao\Data\WPC\ZMD'
with open(path, 'rb') as f:
    data = pickle.load(f)
with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'rb') as f:
    thresRes = pickle.load(f)

for (k,v) in thresRes.items():
    newRes['data'][k]= data[k]
    newRes['thres'][k] = thresRes[k]

with open(r'D:\YANG Luoxiao\Data\WPC\newRes', 'wb') as f:
    pickle.dump(newRes,f)





























