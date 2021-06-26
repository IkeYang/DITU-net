
from matplotlib import pyplot as plt
from utlize import DE,varModel,addNoise,ADE

import numpy as np
from multiprocessing import Pool
import multiprocessing
###add above folder path to work path
import sys
sys.path.append('../')
###add above above folder path to work path
sys.path.append('../..')
from CodeUtlize import mkdir

def makeSparse(x,noise,sparseLevel,sparseRate):
    ind = np.where(noise > sparseLevel)[0]
    indC = np.random.choice(ind, replace=False, size=int(len(ind) * sparseRate))
    x[indC] = 0
    noise[indC] = 0

    ind = np.where(noise > 0.985)[0]
    indC = np.random.choice(ind, replace=False, size=int(len(ind) * sparseRate))
    x[indC] = 0
    noise[indC] = 0
    return x,noise

def yCutDownF(x2,noise,yCutDownLevel):
    ind = np.where(noise > yCutDownLevel)[0]
    x2[ind] = 0
    noise[ind] = 0
    return x2, noise


def generatePict(name,model='DE',show=True,save=True,varProjectF=None):
    Continue = True
    modelPool=[DE,ADE]
    modelL=['DE','ADE']
    while Continue:
        try:
            x = np.arange(0, 1, 1 / 500)
            x2 = np.arange(0.1, 0.7, 1 / 500)
            x = np.concatenate((x, x2))
            xP = np.sort(x)

            ####Sparse
            sparse = True
            sparseLevel = 0.5
            prabSpase = 0.3  #abnormal Rate
            sparseRand = np.random.rand()
            sparseLevel = np.random.randint(int(sparseLevel * 100), 100) / 100
            sparseRate = np.random.randint(90, 95) / 100
            if sparse:
                sparse = sparseRand < prabSpase


            #x cut down
            couOffRate = 0.7
            prabcompletionEnd = 0.5
            completionEnd = np.random.rand()
            end = np.random.randint(int(len(xP) * couOffRate), len(xP))

            # y cut down
            prabYcut=0.1  #abnormal Rate
            yCutDownLevel = np.random.randint(int(0.7 * 100), 100) / 100
            yCutDownRand = np.random.rand()
            yCUtDOwn=yCutDownRand<prabYcut


            ##axis extend
            prabYcut = 0.2   #abnormal Rate
            yAxisExtendRand = np.random.rand()
            xAxisExtendRand = np.random.rand()
            yAxisExtend = np.random.randint(0, 10) / 100
            xAxisExtend = np.random.randint(0, 10) / 100
            if yAxisExtendRand>prabYcut:
                yAxisExtend=0
            if xAxisExtendRand>prabYcut:
                xAxisExtend=0

            ####model
            if model == "DE":
                y, derive = DE(xP,mod='random')
            if model == "ADE":
                y, derive = ADE(xP,mod='random')
            if model=='random':
                modelUsed=np.random.choice([0,1],size=1,p=[0.6,0.4])
                # print(modelUsed)
                m=modelPool[modelUsed[0]]
                y, derive = m(xP, mod='random')
                model=modelL[modelUsed[0]]
            f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))

            xD = xP[:end]
            yD = y[:end]
            # derive1 = derive[:end]
            xD2 = xP[end:]
            yD2 = y[end:]
            # derive2 = derive[end:]
            var = varModel(derive,model=model,varProjectF=varProjectF)
            var1 = var[:end]

            lengthN = np.random.randint(5, 20)
            x2, noise = addNoise(xD, yD, var1, typeN="Gaussion", lengthN=lengthN)
            if sparse:
                x2,noise=makeSparse(x2, noise, sparseLevel, sparseRate)
            if yCUtDOwn:
                x2,noise=yCutDownF(x2,noise,yCutDownLevel)
            ax.plot(x2, noise, '.', color='black')

            lengthN = np.random.randint(5, 10)
            x2, noise = addNoise(xD, yD, var1, typeN="zero", lengthN=lengthN)
            ax.plot(x2, noise, '.', color='black')

            stackN = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.2, 0.1])
            lengthN = np.random.randint(5, 20)
            x2, noise = addNoise(xD, yD, var1, typeN="stacked", lengthN=lengthN, stackNum=stackN)
            if sparse:
                x2, noise = makeSparse(x2, noise, sparseLevel, sparseRate=0.95)
            if yCUtDOwn:
                x2,noise=yCutDownF(x2,noise,yCutDownLevel)
            ax.plot(x2, noise, '.', color='black')

            lengthN = np.random.randint(5, 50)
            x2, noise = addNoise(xD, yD, var1, typeN="random", lengthN=lengthN,xAxisExtend=xAxisExtend,yAxisExtend=yAxisExtend)
            ax.plot(x2, noise, '.', color='black')



            if completionEnd < prabcompletionEnd:
                var2 = var[end:]

                x2, noise = addNoise(xD2, yD2, var2, typeN="Gaussion", lengthN=1)
                if sparse:
                    x2, noise = makeSparse(x2, noise, sparseLevel, sparseRate)
                if yCUtDOwn:
                    x2, noise = yCutDownF(x2, noise, yCutDownLevel)
                ax.plot(x2, noise, '.', color='black')

                x2, noise = addNoise(xD2, yD2, var2, typeN="zero", lengthN=2)
                ax.plot(x2, noise, '.', color='black')
                # x2,noise=addNoise(xD2,yD2,var2,typeN="stacked",lengthN=5)
                # ax.plot(x2,noise,'.',color='b')
                x2, noise = addNoise(xD2, yD2, var2, typeN="random", lengthN=5,xAxisExtend=xAxisExtend,yAxisExtend=yAxisExtend)
                ax.plot(x2, noise, '.', color='black')


            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            ax.set_xlim((0, 1+xAxisExtend))
            ax.set_ylim((0, 1+yAxisExtend))
            if show:
                plt.show()
            if save:
                f.savefig(name + 'x.jpg', dpi=100)
            # f.savefig('1.png', dpi=200, bbox_inches='tight')
            plt.close(f)


            #ground Truth
            f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
            ax.set_xlim((0, 1+xAxisExtend))
            ax.set_ylim((0, 1+yAxisExtend))
            ax.plot(xP, y, color='black', linewidth=3)
            ax.axis('off')
            if show:
                plt.show()
            if save:
                f.savefig(name + 'y.jpg', dpi=100)
            # f.savefig('1.png', dpi=200, bbox_inches='tight')
            plt.close(f)
            Continue = False
            plt.close('all')
        except:
            pass





for i in range(4000):

    print('train ',i)
    savePath=r"TrainData"
    mkdir(mkdir)
    name=savePath+r'\\'+'train%d'%(i)
    generatePict(name, model='random', show=False,save=True)
