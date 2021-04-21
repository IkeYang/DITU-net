#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import optimize

## example  y=(x-1)*(x+2)*(x-8
# def f(x):
#     return (x-1)*(x+2)*(x-8)
# def df(x):
#     return (x+2)*(x-8)+(x-1)*(x+2)+(x-1)*(x-8)
#
# root = optimize.newton(f, -1.6, fprime2=df)
# print(root)
#
#
# root = optimize.newton(f, 1.4, fprime2=df)
# print(root)
#
#
# root = optimize.newton(f, 7, fprime2=df)
# print(root)

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

class polyF():
    def __init__(self):
        self.o=12
        self.coefficients=np.array([ 2.16401179e+04, -1.13965335e+05,  2.49999912e+05, -2.89343363e+05,
        1.77761287e+05, -3.89644276e+04, -1.95186667e+04,  1.70104715e+04,
       -5.46292181e+03,  9.15176784e+02, -7.35125578e+01,  2.16901188e+00,
        1.22141684e-02])
f=polyF()
fx=polyFunction(f)
print(fx.df(0))
print(fx.ddf(0))








