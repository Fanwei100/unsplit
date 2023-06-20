import os,cv2

import torch
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.io import read_image
from unsplit.util import get_examples_by_class
from DataSet import getDataAndModels




def SaveNImagesPerCLASS(testset,dataset,imgsize=64,ImgPath="Images/",N=1,classes=10):
    ImgPath=f"{ImgPath}Size_{imgsize}_{imgsize}/"
    os.makedirs(ImgPath,exist_ok=True)
    images = [get_examples_by_class(testset, c, count=N) for c in tqdm(range(classes))]
    # if N==1:images=[images]
    for clas,imgs in tqdm(enumerate(images)):
        for idx, img in tqdm(enumerate(imgs)):
            save_image(img, f'{ImgPath}{dataset}_{clas}_{idx}.png')

def resizeImages(dataset,size=(480,480),ImgPath="Images/"):
    images = [read_image(ImgPath + f).float()/255 for f in os.listdir(ImgPath) if f.startswith(dataset) and "RS" not in f]
    images=[transforms.Resize(size)(img) for img in images]
    for idx,img in tqdm(enumerate(images)):
        save_image(img, f'{ImgPath}{dataset}_{idx}.png')

def addModelNameInImages():
    for dirpath in ["Results/WithoutTraining/","Results/WithTraining/"]:
        files=os.listdir(dirpath)
        fun=lambda x:x.split(".")[0]+"_0.png"
        filesch=list(map(fun,files))
        for old,new in zip(files,filesch):
            os.rename(dirpath+old,dirpath+new)

def calculateScores():
    lossFile = "Losses.csv"
    if not os.path.exists(lossFile):
        with open(lossFile, "w") as f:
            f.write(f'Model,Dataset,SplitLayer,Class Index,MSE,MAE\n')

    with open(lossFile, "a") as f:
        for model,dataset in zip(["MnistNet","MnistNet","CifarNet"],["mnist","f_mnist","cifar"]):
            baseimages = [read_image("Images/" + f).float()/255 for f in sorted(os.listdir("Images")) if f.startswith(dataset)]
            inversimages = [[read_image("Results/" + f).float()/255 for f in sorted(os.listdir("Results")) if
                             f.startswith(model + "_" + dataset + "_" + str(i + 1))] for i in range(6)]

            mse,mae=torch.nn.MSELoss(),torch.nn.L1Loss()
            for sl,invimages in enumerate(inversimages):
               for index,(base,invers) in enumerate(zip(baseimages,invimages)):
                   f.write(f'{model},{dataset},{sl},{index},{mse(base,invers)},{mae(base,invers)}\n')


def AddColumns(filepath="ServerData/Losses.csv"):
    with open(filepath) as f:
        losslist=f.readlines()
    losslist[0]="Train,"+losslist[0]
    for i in range(1,len(losslist)):
        losslist[i]="No,"+losslist[i]
    with open(filepath,"w") as f:
        f.writelines(losslist)




if __name__ == '__main__':
    dataset,imgsize="flower",64
    # for dataset in ["mnist","f_mnist","cifar","food","flower"]:
    for dataset in [ "flower","food"]:
        trainset,testset,client, server, clone=getDataAndModels(dataset,datasize=imgsize)
        SaveNImagesPerCLASS(testset,dataset,imgsize=imgsize,N=2)

    # resizeImages(dataset,32)
    # addModelNameInImages()
    # calculateScores()
