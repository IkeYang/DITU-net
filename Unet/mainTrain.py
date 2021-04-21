#Author:ike yang

from trainUnet import trainUnet

# for setLen in [2000,3000,5000,6000,7000,8000]:
#     dirP = r"D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE1\TrainData"
#     name='DEAndADE'
#     trainUnet(dirP,name,setLen,epochs=20)





#
# for vp in range(9,10):
#     dirP = r"D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE1\TrainDataVP%d"%(vp)
#     name='DEAndADEVP%d'%(vp)
#     trainUnet(dirP,name,setLen=4000,epochs=20)


# for vp in range(9,10):
# vp=0
# dirP = r"D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE2\TrainDataVP%d"%(vp)
# name='DEAndADE2VP%d'%(vp)
# trainUnet(dirP,name,setLen=4000,epochs=20)


vp=0
dirP = r"D:\YANG Luoxiao\Data\WPC\Generate\DEAndADE2\TrainDataVP%dC1"%(vp)
name='DEAndADE2VP%dC0'%(vp)
trainUnet(dirP,name,setLen=4000,epochs=20)










