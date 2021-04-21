from utlize import DE,varModel,addNoise
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
import multiprocessing




def generatePict(name,model='DE',show=True,couOffRate=0.4,save=True):
    Continue = True
    while Continue:
        try:
            x = np.arange(0, 1, 1 / 500)
            x2 = np.arange(0.1, 0.7, 1 / 500)
            x = np.concatenate((x, x2))
            xP = np.sort(x)

            end = len(xP)
            end = np.random.randint(int(len(xP) * couOffRate), len(xP))

            if model == "DE":
                f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
                y, derive = DE(xP,mod='random')
                xD = xP[:end]
                yD = y[:end]
                derive1 = derive[:end]
                xD2 = xP[end:]
                yD2 = y[end:]
                derive2 = derive[end:]
                var = varModel(derive)
                var1 = var[:end]

                lengthN = np.random.randint(5, 20)
                x2, noise = addNoise(xD, yD, var1, typeN="Gaussion", lengthN=lengthN)
                ax.plot(x2, noise, '.', color='black')

                lengthN = np.random.randint(5, 10)
                x2, noise = addNoise(xD, yD, var1, typeN="zero", lengthN=lengthN)
                ax.plot(x2, noise, '.', color='black')

                stackN = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.2, 0.1])
                lengthN = np.random.randint(5, 20)
                x2, noise = addNoise(xD, yD, var1, typeN="stacked", lengthN=lengthN, stackNum=stackN)
                ax.plot(x2, noise, '.', color='black')

                lengthN = np.random.randint(5, 50)
                x2, noise = addNoise(xD, yD, var1, typeN="random", lengthN=lengthN)
                ax.plot(x2, noise, '.', color='black')

                completionEnd = np.random.rand()

                if completionEnd > 0.5:
                    var2 = var[end:]
                    x2, noise = addNoise(xD2, yD2, var2, typeN="Gaussion", lengthN=1)
                    ax.plot(x2, noise, '.', color='black')
                    x2, noise = addNoise(xD2, yD2, var2, typeN="zero", lengthN=2)
                    ax.plot(x2, noise, '.', color='black')
                    # x2,noise=addNoise(xD2,yD2,var2,typeN="stacked",lengthN=5)
                    # ax.plot(x2,noise,'.',color='b')
                    x2, noise = addNoise(xD2, yD2, var2, typeN="random", lengthN=5)
                    ax.plot(x2, noise, '.', color='black')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                ax.set_xlim((0, 1))
                ax.set_ylim((0, 1))
                if show:
                    plt.show()
                if save:
                    f.savefig(name + 'x.jpg', dpi=100)
                # f.savefig('1.png', dpi=200, bbox_inches='tight')
                plt.close(f)

                f, ax = plt.subplots(1, 1, figsize=(2.56, 2.56))
                ax.set_xlim((0, 1))
                ax.set_ylim((0, 1))
                ax.plot(xP, y, color='black', linewidth=3)
                ax.axis('off')
                if show:
                    plt.show()
                if save:
                    f.savefig(name + 'y.jpg', dpi=100)
                # f.savefig('1.png', dpi=200, bbox_inches='tight')
                plt.close(f)
                Continue = False
        except:
            pass






for i in range(2000):

    print(i)
    savePath=r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly"
    name=savePath+r'\\'+'train%d'%(i)
    generatePict(name, model='DE', show=False)


for i in range(400):
    print(i)
    savePath=r"D:\YANG Luoxiao\Data\WPC\Generate\SecondDEOnly"
    name=savePath+r'\\'+'test%d'%(i)
    generatePict(name, model='DE', show=False)
























