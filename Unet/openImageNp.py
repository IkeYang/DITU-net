
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import optimize
import seaborn as sns
from numpy import polyfit, poly1d
from scipy import interpolate
# plt.figure(1)
# plt.plot(x,y)
# plt.plot(x,f3(x))
# plt.show()
# plt.figure(2)
# plt.imshow(xim)
# plt.show()
import seaborn as sns
def heatmapGreyScale(data,xrange,yrange):
    pass

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
        savePath = r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
        name1 = savePath + r'\\' + savename
        f.savefig(name1 + '.jpg', dpi=100)
    plt.show()
    # plt.close(f)
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
def averageOutpt(y,length=50,end=-1):
    return np.mean(y[-length+end:end])
def paramFunc(f,r1=0.19):
    rd=r1
    fx = polyFunction(f)
    pass1=False
    # r1=findZeropoints(f,0.1)
    # r1=0.20 #[6.89704893e-02 8.21190952e+06 4.61222255e-02]]
    # r1=0.15 #[6.89730015e-02 8.01541967e+06 4.61481945e-02]]
    # r1=0.17 # [6.89687739e-02 8.21190952e+06 4.61022718e-02]]
    # r1=0.18 # [6.89687871e-02 8.11364625e+06 4.61018168e-02]]
    # r1=0.19 #[6.89682816e-02 7.97805636e+06 4.60948755e-02]]
    # # r1=0.14 # [6.89728030e-02 7.78156651e+06 4.61474636e-02]]
    # r1=0.1 #
    #r1=auto
    # try:
    #     r1 = findZeropoints(f,0.15)#0.95 too big?
    # except:
    #     r1 = findZeropoints(f, 0.2)
    # x1=0.15
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
            # x1=optimize.newton(fx.df, r1, fprime2=fx.dddf,fprime=fx.ddf,tol=1e-5,maxiter=20)
            pass1=True
            if f_x:
                if x1>r1 or f(x1)>0.01:
                # if f(x1)>0.01:
                    r1 -= 0.005
                    pass1 = False
                    if r1<0.08:
                        f_x=False
                        r1=0.19
            else:
                # pass
                if x1>r1:
                    r1 -= 0.005
                    pass1 = False
        except:
            r1-=0.005
    # print(2)
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
            # x2 = optimize.newton(fx.df, r2, fprime=fx.ddf,fprime2=fx.dddf)
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
        # return np.array(list(map(self.forward, x)))
        # return np.fromiter((self.forward(xi) for xi in x), x.dtype)

def findR1(x,yy):
    try:
        ind=np.where(yy<0.006)[0][-1]
        return x[ind]
    except:
        return 0.19
def findZeropoints(f,b,c=0.8):
    def pickUpFunc(x):
        return f(x)-b
    return optimize.bisect(pickUpFunc, 0, c)
# def imageToLine(x,p=12,domainCorrlation=True):
#     '''
#
#     :param x: gray pict 256,256
#     :return: x,y
#     '''
#     x = x[31:228, 32:230]
#     xim=np.copy(x)
#     L=x.shape[1]
#     Y=x.shape[0]
#     xL=np.arange(0,1,1/L)
#     y=np.copy(xL)
#     sns.heatmap
#     for i in range(L):
#         coloum=x[:,i]
#         vm = np.min(coloum)
#         lenVM=len(np.where(coloum==vm)[0])
#         if vm>220:
#             y[i]=np.nan
#         else:
#             if lenVM==1:
#                 y[i]=1-np.where(coloum==vm)[0]/Y
#             else:
#                 y[i] = 1 - np.mean(np.where(coloum == vm)[0] )/ Y
#             # else:
#             #     y[i] = 1 - np.mean(np.where(coloum == vm)[0]) / 256
#
#
#     a=np.isnan(y)
#     x=xL[~a]
#     y=y[~a]
#     x[-1]=1
#     # f3 = interpolate.interp1d(x, y, kind='linear')
#     # p=18
#     # try:
#     #     with open('xy', 'rb') as f:
#     #         (xL2,yL2) = pickle.load(f)
#     # except:
#     #     xL2=[]
#     #     yL2=[]
#     # xL2.append(x)
#     # yL2.append(y)
#     # with open('xy', 'wb') as f:
#     #     pickle.dump((xL2,yL2), f)
#     yy=np.copy(y)
#     if not domainCorrlation:
#         f3 = poly1d(polyfit(x,y, p))
#
#         return x,f3(x),f3
#     else:
#         f3 = poly1d(polyfit(x, y, p))
#         xxx=f3(x)
#         vy = averageOutpt(y, length=60, end=-10)
#         # vy = averageOutpt(y, length=int(len( y )*0.25), end=-int(len( y )*0.4))
#         fx = polyFunction(f3)
#
#         pass1 = False
#         try:
#             r2 = findZeropoints(f3,0.95)
#         except:
#             r2 = findZeropoints(f3, 0.85)
#
#         while not pass1:
#             try:
#                 root = optimize.newton(fx.df, r2, fprime2=fx.ddf)
#                 pass1 = True
#                 if root < r2:
#                     r2 += 0.005
#                     pass1 = False
#             except:
#                 r2 += 0.005
#         # y[np.where(x > root * 0.9)[0]] = vy
#         y[np.where(x > root )[0]] = vy
#
#         f3 = poly1d(polyfit(x, y, 12))
#
#         x1, b1, x2, b2=paramFunc(f3)
#         fxx = finalFunc(f3, x1, b1, x2, b2)
#         return x, fxx.forwardVector(x), fxx.forwardVector
#
#     # plt.plot(xL[~a],y[~a])
#     # plt.show()
#     # print(1)
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
        savePath = r"D:\YANG Luoxiao\Data\WPC\Generate\FirstDEOnly"
        name1 = savePath + r'\\' + savename
        f.savefig(name1 + '.jpg', dpi=100)
    plt.show()
    # plt.close(f)
# plotWPCGrey((x,yy),savename='ZMS4WT2Orgin',typeWPC='else')
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
    # f3 = interpolate.interp1d(x, y, kind='linear')
    # p=18
    # try:
    #     with open('xy', 'rb') as f:
    #         (xL2,yL2) = pickle.load(f)
    # except:
    #     xL2=[]
    #     yL2=[]
    # xL2.append(x)
    # yL2.append(y)
    # with open('xy', 'wb') as f:
    #     pickle.dump((xL2,yL2), f)
    yy = np.copy(y)
    if not domainCorrlation:
        f3 = poly1d(polyfit(x, y, p))

        return x, f3(x), f3
    else:
        # f3 = poly1d(polyfit(x, y, p))
        # xxx = f3(x)
        # vy = averageOutpt(y, length=60, end=-10)
        # # vy = averageOutpt(y, length=int(len( y )*0.25), end=-int(len( y )*0.4))
        # fx = polyFunction(f3)
        #
        # pass1 = False
        # try:
        #     r2 = findZeropoints(f3, 0.95)
        # except:
        #     r2 = findZeropoints(f3, 0.85)
        #
        # while not pass1:
        #     try:
        #         root = optimize.newton(fx.df, r2, fprime2=fx.ddf)
        #         pass1 = True
        #         if root < r2:
        #             r2 += 0.005
        #             pass1 = False
        #     except:
        #         r2 += 0.005
        # # y[np.where(x > root * 0.9)[0]] = vy
        # y[np.where(x > root)[0]] = vy
        # r1=findR1(x, yy)
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

        # plt.plot(xL[~a],y[~a])
        # plt.show()
        # print(1)
# def averageOutpt(y,length=50,end=-1):
#     return np.mean(y[-length+end:end])
# plt.figure(1)
# plt.plot(x,y)
# plt.plot(x,f3(x))
# plt.axhline(y=averageOutpt(y,length=50,end=-10), color='r', linestyle='-')
# plt.show()
# plt.figure(2)
# plt.imshow(xim)
# plt.show
# fx=polyFunction(f3)
# print(optimize.newton(fx.df, 0.5, fprime2=fx.ddf))
if __name__=='__main__':
    from numpy import polyfit, poly1d
    name='jf78Ndata22res'
    img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(name)
    with open(img_nameX, 'rb') as f:
        xO1=pickle.load(f)
    # x=x[31:228,32:230]
    x1, y1=imageToLine(xO1)
    # f1=poly1d(polyfit(x1, y1, 9))
    # f2=poly1d(polyfit(x1, y1, 16))
    f3=poly1d(polyfit(x1, y1, 25))
    y2=f3(x1)
    plt.plot(x1,y1,'.')
    # plt.plot(x1,f1(x1))
    # plt.plot(x1,f2(x1))
    plt.plot(x1,f3(x1))

    name = 'jf78Ndata42res'
    img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp" % (name)
    with open(img_nameX, 'rb') as f:
        xO1 = pickle.load(f)
    # x=x[31:228,32:230]
    x1, y1 = imageToLine(xO1)
    # f1=poly1d(polyfit(x1, y1, 9))
    # f2=poly1d(polyfit(x1, y1, 16))
    f3 = poly1d(polyfit(x1, y1, 25))
    y2 = f3(x1)
    plt.plot(x1, y1, '.')
    # plt.plot(x1,f1(x1))
    # plt.plot(x1,f2(x1))
    plt.plot(x1, f3(x1))
    plt.show()
    # name='jf78Ndata4res'
    # img_nameX = r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly\testRes\%sNp"%(name)
    # with open(img_nameX, 'rb') as f:
    #     xO4=pickle.load(f)
    # # x=x[31:228,32:230]
    # x4, y4=imageToLine(xO4)
    # plt.plot(x4,y4)
    # plt.show()
    # print(1)
