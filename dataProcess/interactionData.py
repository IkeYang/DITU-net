
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
import os
def savexy(xy):
    with open('xy','wb') as f:
        pickle.dump(xy,f)

def loadxy():
    with open('xy','rb') as f:
        xy=pickle.load(f)
    return xy

def on_press(event):
    if event.button==3:
        ax.scatter(event.xdata, event.ydata)
        plt.plot([event.xdata, event.xdata], [event.ydata, 600])
        plt.plot([event.xdata, 0], [event.ydata, event.ydata])
        fig.canvas.draw()
    elif event.button==1:
        xy=loadxy()
        xy = np.vstack((xy,np.array([event.xdata, event.ydata]).reshape((1,2))))
        savexy(xy)
        print("x,y=",event.xdata, event.ydata)
        ax.scatter(event.xdata, event.ydata)
        plt.plot([event.xdata, event.xdata], [event.ydata, 0])
        plt.plot([event.xdata, maxX], [event.ydata, event.ydata])
        fig.canvas.draw()



if __name__ == "__main__":
    thresRes = {}
    with open(r'D:\YANG Luoxiao\Data\WPC\ZMD','rb') as f:
        res=pickle.load(f)
    start=1
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresRes', 'rb') as f:
    #     thresRes = pickle.load(f)
    thresRes={}
    for (k, v) in res.items():
        # if k in thresRes.keys():
        #     continue
        if start==0:
            xy=loadxy()
            thresRes[pk]=xy
            # with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'wb') as f:
            #     pickle.dump(thresRes, f)
        print(k)
        data=v
        xy=np.zeros([1,2])
        savexy(xy)
        maxX=max(data[:,0])
        fig = plt.figure()
        fig.canvas.mpl_connect("button_press_event", on_press)
        ax = fig.add_subplot(111)
        ax.plot(data[:,0],data[:,1],'.')
        plt.show()
        start=0
        pk=k

    xy = loadxy()
    thresRes[pk] = xy
    print(thresRes)
    # with open(r'D:\YANG Luoxiao\Data\WPC\thresResZMD', 'wb') as f:
    #     pickle.dump(thresRes, f)
























