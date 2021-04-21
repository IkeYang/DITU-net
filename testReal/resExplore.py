import numpy as np
import pickle
import matplotlib.pyplot as plt



# with open('resZM', 'rb') as f:
#     comparedZM,seqZM,rateZM=pickle.load(f)
# rateZM=np.array(rateZM)
# improvedZM1=comparedZM[:,0,0]-comparedZM[:,1,0]
# improvedZM2=comparedZM[:,0,1]-comparedZM[:,1,1]
# improvedZM3=comparedZM[:,0,2]-comparedZM[:,1,2]
#
# f,axs=plt.subplots(2,3)
# axs[0,0].plot(rateZM[:,0],improvedZM1,'*')
# axs[0,1].plot(rateZM[:,0],improvedZM2,'*')
# axs[0,2].plot(rateZM[:,0],improvedZM3,'*')
#
# axs[1,0].plot(rateZM[:,1],improvedZM1,'*')
# axs[1,1].plot(rateZM[:,1],improvedZM2,'*')
# axs[1,2].plot(rateZM[:,1],improvedZM3,'*')
# plt.show()
#
# with open('res1', 'rb') as f:
#     compared1,seq1,rate1=pickle.load(f)
# rate1=np.array(rate1)
# improved1=compared1[:,0,0]-compared1[:,1,0]
# improved2=compared1[:,0,1]-compared1[:,1,1]
# improved3=compared1[:,0,2]-compared1[:,1,2]
# f,axs=plt.subplots(2,3)
# axs[0,0].plot(rate1[:,0],improved1,'*')
# axs[0,1].plot(rate1[:,0],improved2,'*')
# axs[0,2].plot(rate1[:,0],improved3,'*')
#
# axs[1,0].plot(rate1[:,1],improved1,'*')
# axs[1,1].plot(rate1[:,1],improved2,'*')
# axs[1,2].plot(rate1[:,1],improved3,'*')
# # plt.show()
# print(1)
#
#
# ZMr1r2=[]
# ZMR1r2=[]
# ZMR1R2=[]
# ZMr1R2=[]
# zmr1=0.985
# zmr2=0.9
# zmInd1=[]
# zmInd2=[]
# for (i,k) in enumerate(seqZM):
#     r1=rateZM[i,0]
#     r2=rateZM[i,1]
#     if r1>zmr1:
#         if r2>zmr2:
#             ZMR1R2.append(k)
#             zmInd1.append(i)
#         else:
#             ZMR1r2.append(k)
#     else:
#         if r2 > zmr2:
#             ZMr1R2.append(k)
#         else:
#             ZMr1r2.append(k)
#             zmInd2.append(i)
# print('both bigger',ZMR1R2)
# print('r1 bigger',ZMR1r2)
# print('r2 bigger',ZMr1R2)
# print('both smaller',ZMr1r2)
#
#
# print(np.mean(comparedZM[zmInd1,:,:],axis=0))
# print(np.mean(comparedZM[zmInd2,:,:],axis=0))
#
#
#
#
# r1r2=[]
# R1r2=[]
# R1R2=[]
# r1R2=[]
# zmr1=0.834
# zmr2=0.824
# # zmr1=0.985
# # zmr2=0.9
# removeList=[9,15,17,19,43,65]
# Ind1=[]
# Ind2=[]
# for (i,k) in enumerate(seq1):
#     if i in removeList:
#         print(k)
#         continue
#     r1=rate1[i,0]
#     r2=rate1[i,1]
#     if r1>zmr1:
#         if r2>zmr2:
#             R1R2.append(k)
#             Ind1.append(i)
#         else:
#             r1r2.append(k)
#     else:
#         r1r2.append(k)
#         Ind2.append(i)
# print('both bigger',R1R2)
# print('r1 bigger',R1r2)
# print('r2 bigger',r1R2)
# print('both smaller',r1r2)
#
#
# a=compared1[Ind1,:,:]
# b=compared1[Ind2,:,:]
#
# extract=compared1[:,0,:]-compared1[:,1,:]
#
#
#
#
# print(np.mean(compared1[Ind1,:,:],axis=0))
# print(np.mean(compared1[Ind2,:,:],axis=0))
#
# total=np.concatenate((compared1[Ind1,:,:],comparedZM[zmInd1,:,:]),axis=0)
# total2=np.concatenate((compared1[Ind2,:,:],comparedZM[zmInd2,:,:]),axis=0)
#
# print(np.mean(total,axis=0))
# print(np.mean(total2,axis=0))









with open('resNew', 'rb') as f:
# with open('resZM', 'rb') as f:
    comparedZM,seqZM,rateZM=pickle.load(f)
print(seqZM)
rateZM=np.array(rateZM)
improvedZM1=comparedZM[:,0,0]-comparedZM[:,1,0]
improvedZM2=comparedZM[:,0,1]-comparedZM[:,1,1]
improvedZM3=comparedZM[:,0,2]-comparedZM[:,1,2]

f,axs=plt.subplots(2,3)
axs[0,0].plot(rateZM[:,0],improvedZM1,'*')
axs[0,1].plot(rateZM[:,0],improvedZM2,'*')
axs[0,2].plot(rateZM[:,0],improvedZM3,'*')

axs[1,0].plot(rateZM[:,1],improvedZM1,'*')
axs[1,1].plot(rateZM[:,1],improvedZM2,'*')
axs[1,2].plot(rateZM[:,1],improvedZM3,'*')
plt.show()
extract=comparedZM[:,0,:]-comparedZM[:,1,:]
print(1)





ZMr1r2=[]
ZMR1r2=[]
ZMR1R2=[]
ZMr1R2=[]
zmr1=0.985
zmr2=0.9
zmInd1=[]
zmInd2=[]
zmInd3=[]
for (i,k) in enumerate(seqZM):
    r1=rateZM[i,0]
    r2=rateZM[i,1]
    if r1>zmr1:
        if r2>zmr2:
            ZMR1R2.append(k)
            zmInd1.append(i)
        else:
            ZMR1r2.append(k)
    else:
        if r2 > zmr2:
            ZMr1R2.append(k)
            zmInd1.append(i)
        else:
            ZMr1r2.append(k)
            zmInd2.append(i)
print('both bigger',ZMR1R2)
print('r1 bigger',ZMR1r2)
print('r2 bigger',ZMr1R2)
print('both smaller',ZMr1r2)


print(np.mean(comparedZM[zmInd1,:,:],axis=0))
print(np.mean(comparedZM[zmInd2,:,:],axis=0))
# print(np.mean(comparedZM[zmInd3,:,:],axis=0))
#












