import torch,os
from torchvision.io import read_image
from unsplit.models import *

modelpath="Models_32_32/"
split_layer=2

server=torch.load(modelpath+"Server_mnist_"+str(split_layer)+".pt")
client=torch.load(modelpath+"Client_mnist_"+str(split_layer)+".pt")

imagefolderpath="results/"
files=[imagefolderpath+f for f in os.listdir(imagefolderpath)]
imgs=torch.stack([read_image(file) for file in files]).float()
images=torch.mean(imgs,dim=1,keepdim=True)
# images=imgs[:,0:1]
pred = server(client(images, end=split_layer), start=split_layer + 1)
pred=torch.argmax(pred,dim=1)
print(pred)


