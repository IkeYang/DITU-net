
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
import os



with open(r'D:\YANG Luoxiao\Data\WPC\res','rb') as f:
    res=pickle.load(f)
#
#
# for (k,v) in res.items():
#     print(k)

thresRes={}
### start clear
print('start clear')

with open(r'D:\YANG Luoxiao\Data\WPC\resCoord', 'rb') as f:
    thresRes=pickle.load(f)

for (k,v) in res.items():
    if k in thresRes.keys():
        continue
    print(k)
    plt.plot(v[:,0],v[:,1],'.')
    plt.show()
    num = input('Input the number of clear:')
    xy=np.zeros([int(num),2])
    for i in range(int(num)):
        ws = input("Input the wind speed threshold:")
        power = input("Input the wind power threshold:")
        xy[i,0]=float(ws)
        xy[i,1]=float(power)
    print(k,' coordinate ',xy)
    thresRes[k]=xy

    with open(r'D:\YANG Luoxiao\Data\WPC\resCoord','wb') as f:
        pickle.dump(thresRes,f)

# num=input('Input the number of clear:')
# print(num)
# for i in  range(int(num)):
#     ws=input("Input the wind speed threshold:")
#     power=input("Input the wind power threshold:")
#     print(ws,power)