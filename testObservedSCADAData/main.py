#Author:ike yang

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from testObservedSCADAData.testPic import pipLineTestUnet
# wind speed, wind power
if __name__=='__main__':
    #load data with form [n,[wind speed, wind power]]
    data=np.loadtxt('..\SCADAData\sample.csv', delimiter=",")
    #delete NAN
    data = data[~np.isnan(data[:, 0]), :]
    data = data[~np.isnan(data[:, 1]), :]

    #normalize
    scaler = preprocessing.MinMaxScaler()
    data=scaler.fit_transform(data)

    ##data Must be dict form which ensures the model to test a group of WTs simultaneously
    data={'sample':data}

    #fwpcL is a list containing fwpc sequentially
    fwpcL=pipLineTestUnet(data)

    #display result
    x=np.arange(0,100)/100
    plt.plot(data['sample'][:,0],data['sample'][:,1],'*',label='Observed SCADA Data')
    plt.plot(x,fwpcL[0](x),lw=2,label='WPC')
    plt.legend()
    plt.show()
























