from matplotlib import pyplot as plt
import numpy as np



def varProjectF0(x):
    y=np.zeros_like(x)
    ind=x<0.7
    y[ind]=x[ind]
    y[~ind]=(1-(x[~ind]-1)**4)**0.55
    return y

def DE(x,t1=20,t2=-5,mod='random'):
    while True:
        if mod=='random':
            t1=np.random.uniform(10, 50, 1)
            t2=np.random.uniform(-15, -8, 1)
        y=np.exp(-t1*np.exp(t2*x))
        if np.exp(-t1*np.exp(t2*0.75))>0.95:
            derive = y * (-t1) * np.exp(t2 * x) * t2
            return y,derive



def ADE(x,a1=-10,a2=25,mod='random',a0=5,a3=15):
    while True:
        if mod=='random':
            a1=np.random.uniform(-20, 20, 1)
            a2=np.random.uniform(-15, 10, 1)

        y=np.exp(-np.exp(a0-a1*x-a2*x**2-a3*x**3 ))
        if np.exp(-np.exp(a0-a1*0.7-a2*0.7**2-a3*0.7**3 ))>0.95:
            derive=y*(a1+2*a2*x**1+3*a3*x**2 )*np.exp(a0-a1*x-a2*x**2-a3*x**3 )
            return y,derive

def PLF4II(x,a,b,r,d,mod='random'):
   y=a+(b-a)/(1+(x/r)**d)
   return y

def PLF5(x,a,b,r,d,t,mod='random'):
   y=a+(b-a)/(1+(x/r)**d)**t
   return y





def varModel(derive,model='DE',varProjectF=None):
    if varProjectF is None:
        varProjectF=varProjectF0
    if model=='ADE':
        maxVar = np.random.uniform(0.1, 0.3, 1)
        std =varProjectF((derive - np.min(derive)) / (np.max(derive) - np.min(derive)) ) * maxVar + 0.008
        return std
    maxVar= np.random.uniform(0.1, 0.3, 1)
    std=varProjectF((derive - np.min(derive)) / (np.max(derive) - np.min(derive)) ) * maxVar+0.007
    return std

def addNoise(x,y,std,lengthN=10,typeN="Gaussion",stackNum=1,xAxisExtend=0,yAxisExtend=0):
    if typeN=="Gaussion":
        num1=len(x)
        std=std.reshape((-1,1))
        noise=np.zeros([num1,lengthN])+y.reshape((-1,1))
        randomG=np.random.randn(num1,lengthN)*std
        noise+=randomG
        noise=noise.T.reshape((-1,))
        noise[np.where(noise>1)]=1
        noise[np.where(noise<0)]=0
        x2=np.tile(x, lengthN)
        return x2,noise
    if typeN=="zero":
        num1 = len(x)
        start=np.random.randint(0,int(num1/2))
        end=np.random.randint(start,int(num1*0.9))
        # print(start,end)
        num1 = len(x[start:end])
        noise = np.abs(np.random.randn(num1,lengthN)*0.005)
        x2 = np.tile(x[start:end], lengthN)
        noise = noise.T.reshape((-1,))
        return x2,noise
    if typeN=="stacked":

        ind = np.random.choice(np.where((y > 0.1) & (y < 0.7))[0])
        lengthN=len(x)-ind
        end=np.random.randint(int(1/3*lengthN),lengthN)
        xind = x[ind:ind+end]
        num1 = len(xind)
        var = np.random.uniform(0.001, 0.005, (num1, 1))
        x2 = np.tile(xind, lengthN)
        noise =np.ones([num1, lengthN])*y[ind]+np.random.randn(num1,lengthN)*var
        # stackNum=2
        noise = noise.T.reshape((-1,1))
        for i in range(stackNum-1):

            ind = np.random.choice(np.where((y > 0.1) & (y < 0.9))[0])
            lengthN = len(x) - ind
            end = np.random.randint(int(1 / 3 * lengthN), lengthN)
            xind = x[ind:ind+end]
            num1 = len(xind)
            x3 = np.tile(xind, lengthN)
            var = np.random.uniform(0.002, 0.005, (num1, 1))
            noise2 = np.ones([num1, lengthN]) * y[ind] + np.random.randn(num1, lengthN) * var
            noise2 = noise2.T.reshape((-1,1))
            x2=np.vstack((x2.reshape((-1,1)),x3.reshape((-1,1)))).flatten()
            noise=np.vstack((noise,noise2))
        noise=noise.flatten()
        noise[np.where(noise>1)]=1
        noise[np.where(noise<0)]=0
        return x2,noise
    if typeN=="random":
        # lengthN=np.random.randint(20,50)
        x2=np.random.uniform(0,1+xAxisExtend,(lengthN,))
        noise=np.random.uniform(0,1+yAxisExtend,(lengthN,))
        return x2,noise

def addNoise2(x,y,std,lengthN=10,typeN="Gaussion",stackNum=1,xAxisExtend=0,yAxisExtend=0):
    if typeN=="Gaussion":
        num1=len(x)
        std=std.reshape((-1,1))
        noise=np.zeros([num1,lengthN])+y.reshape((-1,1))
        randomG=np.random.randn(num1,lengthN)*std
        noise+=randomG
        noise=noise.T.reshape((-1,))
        noise[np.where(noise>(1+yAxisExtend))]=1+yAxisExtend
        noise[np.where(noise<0)]=0
        x2=np.tile(x, lengthN)
        return x2,noise
    if typeN=="zero":
        num1 = len(x)
        start=np.random.randint(0,int(num1/2))
        end=np.random.randint(start,int(num1*0.9))
        # print(start,end)
        num1 = len(x[start:end])
        noise = np.abs(np.random.randn(num1,lengthN)*0.005)
        x2 = np.tile(x[start:end], lengthN)
        noise = noise.T.reshape((-1,))
        return x2,noise
    if typeN=="stacked":

        ind = np.random.choice(np.where((y > 0.1) & (y < 0.7))[0])
        lengthN=len(x)-ind
        end=np.random.randint(int(1/3*lengthN),lengthN)
        xind = x[ind:ind+end]
        num1 = len(xind)
        var = np.random.uniform(0.001, 0.005, (num1, 1))
        x2 = np.tile(xind, lengthN)
        noise =np.ones([num1, lengthN])*y[ind]+np.random.randn(num1,lengthN)*var
        # stackNum=2
        noise = noise.T.reshape((-1,1))
        for i in range(stackNum-1):

            ind = np.random.choice(np.where((y > 0.1) & (y < 0.9))[0])
            lengthN = len(x) - ind
            end = np.random.randint(int(1 / 3 * lengthN), lengthN)
            xind = x[ind:ind+end]
            num1 = len(xind)
            x3 = np.tile(xind, lengthN)
            var = np.random.uniform(0.002, 0.005, (num1, 1))
            noise2 = np.ones([num1, lengthN]) * y[ind] + np.random.randn(num1, lengthN) * var
            noise2 = noise2.T.reshape((-1,1))
            x2=np.vstack((x2.reshape((-1,1)),x3.reshape((-1,1)))).flatten()
            noise=np.vstack((noise,noise2))
        noise=noise.flatten()
        noise[np.where(noise>(1+yAxisExtend))]=1+yAxisExtend
        noise[np.where(noise<0)]=0
        return x2,noise
    if typeN=="random":
        # lengthN=np.random.randint(20,50)
        x2=np.random.uniform(0,1+xAxisExtend,(lengthN,))
        noise=np.random.uniform(0,1+yAxisExtend,(lengthN,))
        return x2,noise
if __name__=='__main__':
    while 1:
        x = np.arange(0, 1, 1 / 500)
        y,d=ADE(x)
        f,axs=plt.subplots(1,2)
        axs[0].plot(x,y)
        axs[1].plot(x,d)
        plt.show()




