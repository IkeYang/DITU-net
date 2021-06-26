#Author:ike yang
from trainUnet import trainUnet
dirP = r"..\SCADA Data Synthesis\TrainDataVP0C1"
name='DEAndADE2VP0C0'
trainUnet(dirP,name,setLen=4000,epochs=20)










