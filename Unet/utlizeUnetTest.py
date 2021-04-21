#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import optimize
from numpy import polyfit, poly1d

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
def averageOutpt(y,length=50,end=-1):
    return np.mean(y[-length+end:end])
def paramFunc(f,y):
    fx = polyFunction(f)
    b1=0
    pass1=False
    r1=0.25
    while not pass1:
        try:
            x1=optimize.newton(fx.df, r1, fprime2=fx.ddf)
            pass1=True
        except:
            r1-=0.01
    b1 = f(x1)
    b2=averageOutpt(y,length=50,end=-10)

    pass1 = False
    r2 = 0.5
    while not pass1:
        try:
            x2 = optimize.newton(fx.df, r2, fprime2=fx.ddf)
            pass1 = True
        except:
            r2 += 0.01

    return x1,b1,x2,b2
class finalFunc():
    def __init__(self,f,x1,b1,x2,b2):
        self.f=f
        self.x1=x1
        self.b1=b1
        self.x2=x2
        self.b2=b2

    def forwardVector(self,x):
        yy=self.f(x)
        ind=np.where(x<self.x1)
        yy[ind]*=0
        yy[ind]+=self.b1

        ind = np.where(x > self.x2)
        yy[ind] *= 0
        yy[ind] += self.b2
        return yy.flatten()
with open('xy', 'rb') as f:
    (xL2,yL2) = pickle.load(f)
i=1
x=xL2[i]
y=yL2[i]
f3 = poly1d(polyfit(x,y, 12))
# plt.figure(1)
# plt.plot(x,y)
# plt.plot(x,f3(x))
vy=averageOutpt(y,length=50,end=-10)
# plt.axhline(y=vy, color='r', linestyle='-')
# plt.show()
fx = polyFunction(f3)
root=optimize.newton(fx.df, 0.45, fprime2=fx.ddf)
print(root)
y[np.where(x>root*0.9)[0]]=vy
f3 = poly1d(polyfit(x,y, 12))
# plt.figure(2)
# plt.plot(x,y)
# plt.plot(x,f3(x))
# plt.axhline(y=vy, color='r', linestyle='-')
# plt.show()
fx = polyFunction(f3)
x1=optimize.newton(fx.df, 0.25, fprime2=fx.ddf)
x2=optimize.newton(fx.df, 0.45, fprime2=fx.ddf)
b1=f3(x1)
b2=f3(x2)
# print(x1,b1)
# print(x2,b2)
fxx=finalFunc(f3,x1,b1,x2,b2)
plt.figure(3)
plt.plot(x,y)
plt.plot(x,fxx.forwardVector(x))
plt.axhline(y=vy, color='r', linestyle='-')
plt.show()










































