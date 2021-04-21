###add above folder path to work path
import sys
sys.path.append('../')
###add above above folder path to work path
sys.path.append('../..')
from CodeUtlize import mkdir
from test import p2
def P(x):
    print(x+1)


def proc(x,func=P):
    func(x)


def myP(x):
    print(x**x)
p2()
proc(2,myP)
proc(2)








