
from numpy import polyfit, poly1d
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import optimize
class polyFunction():
    def __init__(self,f):
        self.f=f
        self.order=f.o
        self.coefficients=f.coefficients
    def df(self,x):
        res=0
        for i in range(self.order):
            res+=x**(self.order-i-1)*self.coefficients[i]*(self.order-i)
        return res
    def ddf(self,x):
        res = 0
        for i in range(self.order -1 ):
            res += x ** (self.order - i - 2) * self.coefficients[i]*(self.order-i-1)*(self.order-i)
        return res
    def dddf(self,x):
        res = 0
        for i in range(self.order -2 ):
            res += x ** (self.order - i - 3) * self.coefficients[i]*(self.order-i-2)*(self.order-i-1)*(self.order-i)
        return res

def paramFunc(f,r1=0.19):
    rd=r1
    fx = polyFunction(f)
    pass1=False

    f_x=True
    count=0
    firstcome=True
    while not pass1  :
        count+=1
        try:
            # print(1)
            if count>20:
                if firstcome:
                    r1=0.19
                    firstcome=False
                x1 = optimize.newton(fx.df, r1, fprime2=fx.dddf)
            else:

                x1=optimize.newton(fx.df, r1, fprime=fx.ddf,fprime2=fx.dddf)

            pass1=True
            if f_x:
                if x1>r1 or f(x1)>0.01:
                    r1 -= 0.005
                    pass1 = False
                    if r1<0.08:
                        f_x=False
                        r1=0.19
            else:

                if x1>r1:
                    r1 -= 0.005
                    pass1 = False
        except:
            r1-=0.005

    b1 = f(x1)


    pass1 = False
    try:
        r2 = findZeropoints(f,0.95)#0.95 too big?
    except:
        try:
            r2 = findZeropoints(f, 0.85)
        except:
            try:
                r2 = findZeropoints(f, 0.75)
            except:
                try:
                    r2 = findZeropoints(f, 0.65)
                except:
                    r2 = 0.45
    while not pass1:
        try:
            x2 = optimize.newton(fx.df, r2, fprime2=fx.dddf)
            pass1 = True
            if x2< r2:
                r2 += 0.005
                pass1 = False
        except:
            r2 += 0.005
    b2 = f(x2)
    return x1,b1,x2,b2


class finalFunc():
    def __init__(self,f,x1,b1,x2,b2):
        self.f=f
        self.x1=x1
        self.b1=b1
        self.x2=x2
        self.b2=b2
    def forward(self,x):
        if x<self.x1:
            return self.b1
        elif x<self.x2:
            return self.f(x)
        else:
            return self.f(x)
    def forwardVector(self,x):
        if type(x) is not np.ndarray:
            return self.forward(x)
        yy=self.f(x)
        ind=np.where(x<self.x1)

        yy[ind]*=0
        yy[ind]+=self.b1
        ind = np.where(x > self.x2)
        yy[ind] *= 0
        yy[ind] += self.b2
        return yy.flatten()



def findZeropoints(f,b,c=0.8):
    def pickUpFunc(x):
        return f(x)-b
    return optimize.bisect(pickUpFunc, 0, c)

def plotWPCGrey(input,savename=None,typeWPC='function'):


    f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
    if typeWPC == 'function':
        x = np.arange(0, 1, 1 / 1000)
        y = input(x)
        ax.plot(x, y, color='black')
    elif typeWPC == 'curve':
        x = input[0]
        y = input[1]
        ax.plot(x, y, color='black')
    else:
        x = input[0]
        y = input[1]
        ax.plot(x, y,  '.',ms=1.2,color='black')
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if savename is not None:
        savePath = r"savePict"
        name1 = savePath + r'\\' + savename
        f.savefig(name1 + '.jpg', dpi=100)
    plt.show()

def imageToLine(x, p=12, domainCorrlation=True):
    '''

    :param x: gray pict 256,256
    :return: x,y
    '''
    x = x[31:228, 32:230]
    xim = np.copy(x)
    L = x.shape[1]
    Y = x.shape[0]
    xL = np.arange(0, 1, 1 / L)
    # xL = np.arange(0, L)*(1/(L-1))
    y = np.copy(xL)
    y0=0
    for i in range(L):
        coloum = x[:, i]
        vm = np.min(coloum)
        lenVM = len(np.where(coloum == vm)[0])
        if vm > 220:#220
            y[i] = np.nan
        else:
            if lenVM == 1:
                y[i] = 1 - np.where(coloum == vm)[0] / (Y)
            else:
                y[i] = 1 - np.mean(np.where(coloum == vm)[0]) / (Y)
            while y[i] -y0>0.1:
                coloum[np.argmin(coloum)]=224
                vm = np.min(coloum)
                if vm > 220:  # 220
                    y[i] = np.nan
                    break
                else:
                    y[i] = 1 - np.mean(np.where(coloum == vm)[0]) / (Y)

            if y[i] != np.nan:
                y0=y[i]
            # else:
            #     y[i] = 1 - np.mean(np.where(coloum == vm)[0]) / 256

    a = np.isnan(y)
    x = xL[~a]
    y = y[~a]
    x[-1] = 1

    if not domainCorrlation:
        f3 = poly1d(polyfit(x, y, p))

        return x, f3(x), f3
    else:

        f3 = poly1d(polyfit(x, y, 12))
        import time
        s=time.time()
        x1, b1, x2, b2 = paramFunc(f3,0.19)

        fxx = finalFunc(f3, x1, b1, x2, b2)
        if b1<0 and x1>0:
            # print(fxx.forwardVector(0),fxx.forwardVector(0.2))
            x1 = findZeropoints(fxx.forwardVector, 0,0.2)
            b1=fxx.forwardVector(x1)
            fxx = finalFunc(fxx.forwardVector, x1, b1, x2, b2)
        e = time.time()
        # print(e-s)
        return x, fxx.forwardVector(x), fxx.forwardVector


