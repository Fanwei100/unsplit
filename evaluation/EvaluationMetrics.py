import os,math,cv2
import sys
import time
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch,numpy
import torchvision.io
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from ignite.engine import *
from ignite.metrics import *
from image_similarity_measures.quality_metrics import rmse, psnr,sre,ssim,fsim,issm,uiq,sam

if os.getcwd().endswith("evaluation"):os.chdir("../")
# create default evaluator for doctests
def eval_step(engine, batch):
    return batch

class Evaluation:
    def __init__(self,dataset=None,image_size=64,modelname="efficientnet_v2_s",ep="_149_1.000_0.747",useCude=False):
        if dataset is not None:
            self.model = torch.load(f"evaluation/Models/Size_{image_size}_{image_size}/{modelname}_{dataset}{ep}.pt")
            # self.model.classifier = self.model.classifier[:1]
            self.model.eval()
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if useCude else torch.device('cpu')
            print(f"{modelname} {dataset} Model Loaded for size {image_size}")

    def getClass(self,input):
        if next(self.model.parameters()).is_cuda:
            input=input.cuda()
        return self.model(input).cpu()

    @staticmethod
    def caclulateRMSE(preds, target):
        return rmse(preds, target)

    @staticmethod
    def caclulatePNSR(preds, target):
        return psnr(preds, target)

    @staticmethod
    def caclulateSre(preds, target):
        return sre(preds, target)
    @staticmethod
    def caclulateSsim(preds, target):
        return ssim(preds, target)
    @staticmethod
    def caclulateFsim(preds, target):
        return fsim(preds, target)

    @staticmethod
    def caclulateIsim(preds, target):
        return issm(preds, target)
    @staticmethod
    def caclulateUiq(preds, target):
        return uiq(preds, target)

    @staticmethod
    def caclulateSam(preds, target):
        return sam(preds, target)



    # create default trainer for doctests
    # as handlers could be attached to the trainer,
    # each test must define his own trainer using `.. testsetup:`

    def calulateSSIM(self,preds, target):
        default_evaluator = Engine(eval_step)
        metric = SSIM(data_range=1.0)
        metric.attach(default_evaluator, 'ssim')
        state = default_evaluator.run([[preds, target]])
        return state.metrics['ssim']

    def calulateFID(self,preds, target):
        default_evaluator = Engine(eval_step)
        metric = FID(num_features=4096, feature_extractor=self.model)
        metric.attach(default_evaluator, "fid")
        state = default_evaluator.run([[preds, target]])
        return state.metrics["fid"]


    def calculate_fid(self,preds, target):
        # calculate frechet inception distance
        # calculate activations
        with torch.no_grad():
            self.model=self.model.to(self.device)
            act1 = self.model(preds.to(self.device)).cpu().numpy()
            act2 = self.model(target.to(self.device)).cpu().numpy()
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


    def calculatFD(self,y_pred,y_true):
        with torch.no_grad():
            generated = self.model(y_pred.to(self.device)).cpu().numpy()
            real = self.model(y_true.to(self.device)).cpu().numpy()
        fd = numpy.linalg.norm(generated - real)
        return fd

    def calculateScore(self,preds, target):
        claspred=torch.argmax(self.getClass(preds),axis=1).numpy()[0]
        # preds,target=(np.rollaxis(preds.numpy(),1,4)*255).astype(int)[0],(np.rollaxis(target.numpy(),1,4)*255).astype(int)[0]
        preds,target=(np.rollaxis(preds.cpu().numpy(),1,4)*255)[0],(np.rollaxis(target.cpu().numpy(),1,4)*255)[0]
        # fsim=self.caclulateFsim(preds,target)
        ssim=self.caclulateSsim(preds,target)
        return claspred,ssim


def calculateTheScores(datasets,classes,indexes,normalize=False,folders=["WithoutTraining"],realimages="Images/",genetaredimages="Results/",outfilepath="Graphs/Scores"):
    columns="Model,Dataset,SplitPoint,Class,Index,RMSE,PSME,SRE,SSIM,Fsim,Issm,Uiq,Sam\n"
    resdata_list=[]
    for dataset in datasets:
        for folder in folders:
            for cls in classes:
                for ind in tqdm(indexes):
                    generatedImasgePaths=[f"{genetaredimages}{folder}/{model}_{dataset}_{sp}_{cls}_{ind}.png" for model in ["CifarNet","Vgg8Net"] for sp in range(1,7)]
                    realImasgePaths=f"{realimages}{dataset}_{cls}_{ind}.png"
                    realimage=cv2.imread(realImasgePaths)
                    if normalize: realimage=realimage/np.max(realimage)*255
                    for genpath in generatedImasgePaths:
                        if not os.path.exists(genpath): continue
                        genimage=cv2.imread(genpath)
                        if normalize: genimage = genimage / np.max(genimage) * 255
                        if np.any(np.isnan(genimage)):print(genpath)
                        print(np.max(genimage),np.max(realimage), genpath)
                        rmse=Evaluation.caclulateRMSE(genimage,realimage)
                        pnsr=Evaluation.caclulatePNSR(genimage,realimage)
                        sre=Evaluation.caclulateSre(genimage,realimage)
                        ssim=Evaluation.caclulateSsim(genimage,realimage)
                        fsim=Evaluation.caclulateFsim(genimage,realimage)
                        isim=Evaluation.caclulateIsim(genimage,realimage)
                        uiq=Evaluation.caclulateUiq(genimage,realimage)
                        sam=Evaluation.caclulateSam(genimage,realimage)
                        scores=f'{rmse},{pnsr},{sre},{ssim},{fsim},{isim},{uiq},{sam}'
                        resdata_list.append(f"{genpath.split('/')[-1].split('.')[0].replace('_',',')},{scores}\n")
    if normalize: outfilepath+="_Normalize"
    with open(outfilepath+".csv","w") as f:
        f.write(columns)
        f.writelines(resdata_list)

def calculateTheTorchScores(filepath="Graphs/Losses.csv",realimages="Images/",genetaredimages="Results/"):
    df=pd.read_csv(filepath)
    dfgroup=df.groupby("Dataset")
    train_yesNo={"Yes":"WithTraining","No":"WithoutTraining"}
    resultdf=[]
    for dataset,dfgrp in dfgroup:
        eval=Evaluation(dataset)
        for model,dfg in dfgrp.groupby("Model"):
            # indexadd=1-dfg["SplitLayer"].min()
            # dfg["SplitLayer"]=dfg["SplitLayer"].map(lambda x:x+indexadd)
            for index,d in dfg.iterrows():
                realimagepath=realimages+dataset+"_"+str(d["Class"])+".png"
                gneratedimagepath=f"{genetaredimages}{train_yesNo[d['Train']]}/{model}_{dataset}_{d['SplitLayer']}_{d['Class Index']}.png"
                real,generated=torch.unsqueeze(torchvision.io.read_image(realimagepath).float()/255,dim=0),torch.unsqueeze(torchvision.io.read_image(gneratedimagepath).float()/255,dim=0)
                if "mnist" in dataset:
                    real,generated=torch.mean(real,dim=1,keepdim=True),torch.mean(generated,dim=1,keepdim=True)
                ssim=eval.calulateSSIM(generated,real)
                fd=eval.calculatFD(generated,real)
                d["ssim"]=ssim
                d["fd"]=fd
                resultdf.append(d)
    df=pd.concat(resultdf, axis=1).T
    df.to_csv("Graphs/Reports.csv")


def calculateMissingScores(filepath="Losses.csv",realimages="Images/",genetaredimages="Results/"):
    df=pd.read_csv(filepath)
    dfgroup=df.groupby("Dataset")
    train_yesNo={"yes":"WithTraining","No":"WithoutTraining"}
    resultdf=[]
    for dataset,dfgrp in dfgroup:
        eval=Evaluation(dataset)
        for model,dfgr in dfgrp.groupby("Model"):
            for index,dfg in dfgr.groupby("Index"):
                for _,d in dfg.iterrows():
                    realimagepath=f"{realimages}{dataset}_{d['Class']}_{index}.png"
                    generatedimagepath=f"{genetaredimages}{train_yesNo[d['Train']]}/{model}_{dataset}_{d['SplitLayer']}_{d['Class']}_{index}.png"
                    real,generated=torch.unsqueeze(torchvision.io.read_image(realimagepath).float(),dim=0),torch.unsqueeze(torchvision.io.read_image(generatedimagepath).float(),dim=0)
                    real, generated = real/torch.max(real),generated/torch.max(generated)
                    if "mnist" in dataset:
                        real,generated=torch.mean(real,dim=1,keepdim=True),torch.mean(generated,dim=1,keepdim=True)
                    ssim=eval.calulateSSIM(generated,real)
                    fd=eval.calculatFD(generated,real)
                    print(f"{model}_{dataset}_{d['SplitLayer']}_{d['Class']}_{index}",ssim,fd,type(ssim))
                    if not math.isnan(ssim): d["SSIM"]=ssim
                    if not math.isnan(fd): d["FD"]=fd
                    resultdf.append(d)
    df=pd.concat(resultdf, axis=1).T
    print(df.shape)
    print(df[df["SSIM"]=="Na"].shape)
    print(df[df["SSIM"]==np.nan].shape)
    df.to_csv("Losses11.csv",index=None)


def calculateFIDScores(maxindex=2, realimages="Images/", genetaredimages="Results/", resfolder="WithoutTraining/", OutFile="Graphs/Report_Fid.csv"):
    #     for dataset,model in (("mnist","MnistNet"),("f_mnist","MnistNet"),("cifar","CifarNet"),("food","Vgg8Net"),("food","CifarNet"),("flower","Vgg8Net"),("flower","CifarNet")):

    processedRecrd = []
    if not os.path.exists(OutFile):
        with open(OutFile, "w") as f:
            f.write(f'Train,Model,Dataset,SplitLayer,Class,FID\n')
    else:
        with open(OutFile) as f:
            processedRecrd=[",".join(d.split(",")[:5]) for d in f.readlines()[1:]]

    for dataset,model in (("food","Vgg8Net"),("flower","Vgg8Net"),("food","CifarNet"),("flower","CifarNet")):
        eval = Evaluation(dataset)
        remained=0
        for sp in range(1,7):
            print(f"Working for {dataset} dataset with {model} model {sp} splitpoint")
            for cl in tqdm(range(0,10)):
                try:
                    instanceinfo = f'{"yes" if resfolder=="WithTraining" else "No"},{model},{dataset},{sp},{cl}'
                    if instanceinfo in processedRecrd:
                        print(instanceinfo,"already Processed")
                        continue
                    images,results=[torchvision.io.read_image(f"{realimages}{dataset}_{cl}_{i}.png").float() for i in range(maxindex)], [torchvision.io.read_image(f"{genetaredimages}{resfolder}{model}_{dataset}_{sp}_{cl}_{i}.png")  for i in range(maxindex)]
                    images, results=torch.stack([img/torch.max(img) for img in images]),torch.stack([img/torch.max(img) for img in results])
                    if "mnist" in dataset:
                        images, results=torch.mean(images,dim=1,keepdim=True),torch.mean(results,dim=1,keepdim=True)
                    fid=eval.calculate_fid(images,results)
                    with open(OutFile, "a") as f:
                        f.write(f'{instanceinfo},{fid}\n')
                    processedRecrd.append(instanceinfo)
                except Exception as e:
                    remained+=1
                    print("E",e,instanceinfo)
                    print(torch.any(torch.isnan(results[0])),torch.any(torch.isnan(results[1])))
        print(remained,"to be processed")
def checkResults(maxindex=2, realimages="Images/", genetaredimages="Results/", resfolder="WithoutTraining/"):
    for dataset,model in (("flower","CifarNet"),("food","CifarNet"),("flower","CifarNet"),("food","Vgg8Net"),("flower","Vgg8Net")):
        for sp in range(1,7):
            print(f"Working for {dataset} dataset with {model} model {sp} splitpoint")
            for cl in tqdm(range(0,10)):
                instanceinfo = f'{"yes" if resfolder=="WithTraining" else "No"},{model},{dataset},{sp},{cl}'
                images,results=[torchvision.io.read_image(f"{realimages}{dataset}_{cl}_{i}.png").float() for i in range(maxindex)], [torchvision.io.read_image(f"{genetaredimages}{resfolder}{model}_{dataset}_{sp}_{cl}_{i}.png")  for i in range(maxindex)]

def DeleteImagesWithLowerThreashHolds(thrhols=10,folder="Results/"):
    df=pd.read_csv("Losses.csv")
    dflist=[]
    for indx,row in df.iterrows():
        subfolder="WithTraining/" if row["Train"]=="yes" else "WithoutTraining/"
        filepath=f"{folder}{subfolder}{row['Model']}_{row['Dataset']}_{row['SplitLayer']}_{row['Class']}_{row['Index']}.png"
        img=cv2.imread(filepath)
        if np.max(img)>=thrhols:
            dflist.append(row)
    df = pd.concat(dflist, axis=1).T
    df.to_csv("ServerData/Losses.csv")


if __name__ == '__main__':
    eval=Evaluation()
    pred,target=np.random.random((32,32,60)),np.random.random((32,32,60))
    print(eval.caclulateFsim(pred,target))
    print(eval.caclulateSsim(pred,target))
    print(eval.caclulateSam(pred,target))
    print(eval.caclulateUiq(pred,target))
    print(eval.caclulateSre(pred,target))
    print(eval.caclulatePNSR(pred,target))
    print(eval.caclulateIsim(pred,target))
    print(eval.caclulateRMSE(pred,target))
    eval=Evaluation("food")
    # print(eval.calculateScore(torch.abs(torch.randn((1,3,64,64))),torch.abs(torch.randn((1,3,64,64)))))
    # calculateTheScores(["food","flower"],classes=range(0,10),indexes=range(2),normalize=False)
    # calculateMissingScores()
    # calculateFIDScores()
    # checkResults()
    # DeleteImagesWithLowerThreashHolds(200)





