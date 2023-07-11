import os,torch
import io
buffer = io.BytesIO()

from DataSet import getDataAndModels

trainset, testset, clientModel, serverModel, _ = getDataAndModels("food", model="Vgg8Net", datasize= (256,256))

def writeintofile(d,layer=0,out="Results/Intermediate/"):
	os.makedirs(out,exist_ok=True)
	torch.save(d, buffer)
	with open(f"{out}layer_{layer}.txt","w") as f:
		f.write(str([[[k for k in j] for j in i] for i in d.numpy()]))



d=torch.unsqueeze(testset[0][0],dim=0)
# writeintofile(d[0],layer=0)
for i in range(1,7):
	out=clientModel(d,end=i).detach()
	writeintofile(out[0],layer=i)

