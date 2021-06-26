#Author:ike yang
from trainUnet import trainUnet
dirP = r"..\SCADA Data Synthesis\TrainData"
name='DEAndADE'
trainUnet(dirP,name,setLen=4000,epochs=20)










