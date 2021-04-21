import pickle
import numpy as np
import matplotlib.pyplot as plt
amL= ['ZMS1WT7', 'A2-30', 'ZMS1WT2', 'A1-14', 'A2-33', 'A1-12', 'A1-09', 'JF-50', 'A2-22', 'ZMS1WT3', 'ZMS1WT8', 'LN35', 'ZMS1WT1', 'A1-13', 'A1-10', 'A1-11', 'A2-19', 'JF-64', 'LN41', 'ZMS1WT6', 'A1-08', 'A2-21', 'A1-02', 'LN42', 'LN40', 'LN49', 'A1-07', 'LN51', 'LN53', 'LN43', 'A1-05', 'JF-78', 'ZMS1WT4', 'A2-23', 'LN62', 'A2-25', 'LN54', 'LN63', 'ZMS1WT9', 'JF-33']
import seaborn as sns
# font = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 12,
#         }
sns.set(style="white")

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
        }

def returnIndex(seq,L1):
    bL=[]
    mL=[]
    for i,k in enumerate(seq):
        if k in L1:
            mL.append(i)
        else:
            bL.append(i)
    return bL,mL


# ['Unet', 'KNN', 'SNN', 'DE', 'ADE', 'PLF4', 'PLF5', 'Spline']
x=[0.875,0.90,0.925,0.95,0.975,1]
nameL=['resCTotal0-887.5','resCTotal0-890.0','resCTotal0-892.5','resCTotal0-890.0','resCTotal0-897.5','resTotal0-8T']
resU=[]
resU2=[]
resSR=[]
resSNN=[]
resSNN2=[]
SU=2
SS=2
metric=2
for name in nameL:

    with open(name, 'rb') as f:
        res=pickle.load(f)

    print(res['methodOrder'])
    av=np.mean(res['res'],axis=(1))
    # print(av[0,S,metric])
    resU.append(av[0,SU,metric])
    # resSR.append(av[-1,S,metric])
    resSNN.append(av[2,SS,metric])
metric=0
for name in nameL:

    with open(name, 'rb') as f:
        res=pickle.load(f)

    print(res['methodOrder'])
    av=np.mean(res['res'],axis=(1))
    # print(av[0,S,metric])
    resU2.append(av[0,SU,metric])
    # resSR.append(av[-1,S,metric])
    resSNN2.append(av[2,SS,metric])

f, ax = plt.subplots(1, 1)
ax.set_ylabel('RMSE', font)
ax.set_xlabel('Quantile', font)
ax.tick_params(labelsize=7)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lns1=ax.plot(x,resU2,label='STU-net (RMSE)')
lns2=ax.plot(x,resSNN2,label='SNN (RMSE)')



ax2 = ax.twinx()
ax2.set_ylabel('MAE', font)
ax2.set_xlabel('Quantile', font)
ax2.tick_params(labelsize=7)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lns3=ax2.plot(x,resU,'--',label='STU-net (MAE)')
lns4=ax2.plot(x,resSNN,'--',label='SNN (MAE)')


lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc='best', prop=font)

plt.show()
f.savefig(r'savePic/curdownUS.jpg' )
fig = plt.figure()
