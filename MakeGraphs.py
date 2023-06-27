import os,pandas as pd,cv2,torch,PIL
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from torchvision.datasets.vision import VisionDataset
from typing import Any, Tuple
from torchvision import transforms
from sklearn.metrics import accuracy_score
# from evaluation.EvaluationMetrics import Evaluation

# EvaluationMetrices={"RMSE": Evaluation.caclulateRMSE, "PNSR": Evaluation.caclulatePNSR, "SRE": Evaluation.caclulateSre,"SSIM": Evaluation.caclulateSsim,
#         "FSIM": Evaluation.caclulateFsim, "ISIM": Evaluation.caclulateIsim, "UIQ": Evaluation.caclulateUiq}
EvaluationMetrices={}

class Dataset(VisionDataset):
    def __init__(self,folderpath,dataset,outfilepath,train_percent="0",datasize=64) -> None:
        self.transform= transforms.Compose([
                 transforms.Resize(datasize),
                 transforms.ToTensor(),
        ])
        files=os.listdir(folderpath)
        files=list(filter(lambda x:x.split("_")[1]==dataset,files))
        if os.path.exists(outfilepath):
            with open(outfilepath) as f:
                existRecord=["_".join(d[:-1].split(",")[1:-1])+".png" for d in f.readlines()[1:] if d.split(",")[0]==train_percent]
                files=list(filter(lambda x:x not in existRecord,files))
        self._labels=[train_percent+','+f.split(".")[0].replace("_",",") for f in files]
        self._image_files=[os.path.join(folderpath,f) for f in files]

    def __len__(self) -> int:
        return len(self._image_files)
    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



def makeInversionGraph(model,dataset,seed,loss="mse",index=0,fig = None,imgsize=64,trainpercentage=0,EnhanceImages=False,convertColor=False,Classes=10,outdir="Results/Graphs/"):
    traintype = "WithTraining_"+str(trainpercentage)+"/" if trainpercentage>0 else "WithoutTraining/"
    baseimages=[cv2.imread(f"Images/Size_{imgsize}_{imgsize}/{dataset}_{i}_{index}.png") for i in range(Classes)]
    # inversimages = [[cv2.imread("Results/"+traintype + f) for f in sorted(os.listdir("Results/"+traintype)) if f.startswith(model+"_"+dataset+"_"+str(i+1))] for i in range(6)]
    resbasepath=f"Results/Size_{imgsize}_{imgsize}/{traintype}{model}_{dataset}_{loss}"
    inversimages = [[cv2.imread(f"{resbasepath}_{sp}_{cl}_{index}_{seed}.png") for cl in range(Classes) if os.path.exists(f"{resbasepath}_{sp}_{cl}_{index}_{seed}.png")] for sp in range(1,7)]


    if len(inversimages[0])==0: return
    if convertColor:
        baseimages = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in baseimages]
        inversimages=[[cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in invimg] for invimg in inversimages]
    if EnhanceImages:
        inversimages=[[(img.astype(float)/(np.max(img)+1e-8)*255).astype(int) for img in invimg] for invimg in inversimages]


    if fig is None: fig=plt.figure(figsize=(20., 15.))
    grid = ImageGrid(fig,nrows_ncols=(len(inversimages)+1, Classes),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    grid=list(grid)
    for i,im in enumerate(baseimages):
        # Iterating over the grid returns the Axes.
        grid[i].imshow(im)

    for i, imglist in enumerate(inversimages):
        for j,im in enumerate(imglist):
            grid[(i+1)*Classes+j].imshow(im)

    midxsize=inversimages[0][0].shape[0]//2
    midysize=inversimages[0][0].shape[1]//2
    for i in range(Classes):
        grid[i].set_xticks([midxsize])
        grid[i].set_xticklabels([str(i)])

    grid[0].set_yticks([midxsize])
    grid[0].set_yticklabels(["Ref"])

    for i in range(1,len(inversimages)+1):
        grid[i*Classes].set_yticks([midysize])
        grid[i*Classes].set_yticklabels([str(i)])

    for g in grid:
        for tick in g.xaxis.get_major_ticks():
            tick.label.set_fontsize(70)
        for tick in g.yaxis.get_major_ticks():
            tick.label.set_fontsize(70)

    grid[30].set_ylabel("Split Point",fontsize=70)
    grid[64].set_xlabel("Image Index",fontsize=70)
    fig.suptitle(f"Datase {dataset}, Model {model}, Loss {loss}, Index {index} {traintype[:-1]}",fontsize=30)
    outdir=f"{outdir}{dataset}_{model}_{loss}/"
    Path(outdir).mkdir(exist_ok=True,parents=True)
    filepath=f"{outdir}{index}_{seed}_{traintype[:-1]}.png"
    plt.savefig(filepath)
    print("Save at ",filepath)
    return filepath


def makeInversionMetrics():
    convertColor=True
    index=1
    makeInversionGraph("MnistNet","mnist",index=index)
    makeInversionGraph("MnistNet","f_mnist",index=index)
    makeInversionGraph("CifarNet","cifar",index=index)
    makeInversionGraph("Vgg8Net","flower",index=index,convertColor=convertColor)
    makeInversionGraph("CifarNet","flower",index=index,convertColor=convertColor)
    makeInversionGraph("Vgg8Net","food",index=index,convertColor=convertColor)
    makeInversionGraph("CifarNet","food",index=index,convertColor=convertColor)
    plt.show()


def makeMerircsGraph(model,dataset,metrics_list,index=0,trained=False,EnhanceImages=False,convertColor=False,Classes=10):
    traintype = "WithTraining/" if trained else "WithoutTraining/"
    baseimages = [cv2.imread(f"Images/{dataset}_{i}_{index}.png") for i in range(Classes)]
    resbasepath = f"Results/{traintype}{model}_{dataset}"
    inversimages = np.array([[cv2.imread(f"{resbasepath}_{sp}_{cl}_{index}.png") for cl in range(Classes) if os.path.exists(f"{resbasepath}_{sp}_{cl}_{index}.png")] for sp in range(1, 7)])
    figure, axis = plt.subplots(1,len(metrics_list)+1,figsize=(20, 6))
    for i,metrics in enumerate(metrics_list):
        # i,j=i//colmcount,i%colmcount
        metricsvalue= np.array([[EvaluationMetrices[metrics](inv,real) for inv,real in zip(invimg,baseimages)]for invimg in inversimages])
        axis[i].plot(metricsvalue)
        axis[i].set_xlabel("Split Point")
        axis[i].set_ylabel(metrics)
        axis[i].legend(["image"+str(i) for i in range(10)],ncol=4,bbox_to_anchor=(0.95,-0.2))
        axis[i].yaxis.grid()
        axis[i].title.set_text(f"{metrics} for {model} with {dataset} dataset")
    imgpath=makeInversionGraph(model, dataset, index=index, fig=None, trained=trained, EnhanceImages=EnhanceImages, convertColor=convertColor,Classes=Classes)
    # imgpath="Results/Graphs/flower_Vgg8Net_WithoutTraining.png"
    img=cv2.imread(imgpath)
    h,w,_=img.shape
    img=img[int(h*.13):int(h*(1-.13)),int(w*.122):int(w*(1-.1))]
    h,w,_=img.shape
    axis[-1].imshow(img)
    axis[-1].title.set_text(f"{model} with {dataset} dataset")
    axis[-1].set_yticks(np.arange(h//12,h,h//6))
    axis[-1].set_yticklabels([str(i) for i in range(1,7)])
    axis[-1].set_xticks(np.arange(w//20,w,w/10))
    axis[-1].set_xticklabels([str(i) for i in range(10)])
    figure.tight_layout(pad=2.0)
    # figure.suptitle(f"{','.join(metrics_list)} for  {model} with {dataset} dataset",fontsize=30)
    os.makedirs("Results/Graphs/MetricsGraph",exist_ok=True)
    figure.savefig(f"Results/Graphs/MetricsGraph/{'_'.join(metrics_list)}_{dataset}_{model}_{index}_{traintype[:-1]}.png")

def makeMerircGraph(model,dataset,metrics,index=0,trained=False,Classes=10):
    traintype = "WithTraining/" if trained else "WithoutTraining/"
    baseimages = [cv2.imread(f"Images/{dataset}_{i}_{index}.png") for i in range(Classes)]
    resbasepath = f"Results/{traintype}{model}_{dataset}"
    inversimages = np.array([[cv2.imread(f"{resbasepath}_{sp}_{cl}_{index}.png") for cl in range(Classes) if os.path.exists(f"{resbasepath}_{sp}_{cl}_{index}.png")] for sp in range(1, 7)])
    metricsvalue= np.array([[EvaluationMetrices[metrics](inv,real) for inv,real in zip(invimg,baseimages)]for invimg in inversimages])
    figure, axis = plt.subplots(1, 1, figsize=(20, 10))
    axis.plot(metricsvalue)
    axis.set_xlabel("Split Point",fontsize="20")
    axis.set_ylabel(metrics,fontsize="20")
    axis.legend(["image" + str(i) for i in range(10)], ncol=5, bbox_to_anchor=(0.8, -0.2),fontsize="20")
    axis.yaxis.grid()
    # axis.title.set_text(f"{metrics} for {model} with {dataset} dataset")
    plt.title(f"{metrics} for {model} with {dataset} dataset",fontsize=25)
    figure.tight_layout(pad=2.0)
    os.makedirs("Results/Graphs/MetricsGraph",exist_ok=True)
    figure.savefig(f"Results/Graphs/MetricsGraph/{dataset}_{model}_{index}_{traintype[:-1]}_{metrics}.png")

def addErrorBar(plt,mean,std,color,label,plotype="Error"):
    if plotype=="Error":
        plt.errorbar(range(len(mean)), mean, yerr=std,label=label, linestyle='--', marker='p',color=color, markerfacecolor=color,ecolor=color, markersize=10)
    plt.plot(mean,label=label,color=color)

def PlotGraphFor(meanlist,stdlist,colorlist,labellist,ylabel,title,outfile,plotype="Error"):
    for m,s,c,l in zip(meanlist,stdlist,colorlist,labellist):
        addErrorBar(plt,m,s,c,l,plotype=plotype)
    plt.xlabel("Split Point")
    plt.xticks(range(6),[str(i) for i in range(1,7)])
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(outfile)
    plt.close()
    # plt.show()

def makeAverageLossGraph(trainpercentage,dataset,model,loss="mse",imgsize=64,index=0,outGraph="Results/Graphs/AverageLoss/"):

    Path(outGraph).mkdir(exist_ok=True,parents=True)
    losses=pd.read_csv(f"Results/Size_{imgsize}_{imgsize}/CSVFiles/Losses.csv")
    losses=losses.replace("No",0)
    losses["Train"]=losses["Train"].astype(int)
    losses=losses[losses["Dataset"]==dataset]
    losses=losses[losses["Model"]==model]
    # losses=losses[losses["Train"]==trainpercentage]
    losses=losses[losses["Index"]==index]
    losses=losses[losses["LossName"]==loss]
    for cls,clgroup in losses.groupby("Class"):
        msemean,ssimmean,fsimmean,msestd,ssimstd,fsimstd,rounds= {}, {}, {}, {}, {}, {}, {}
        for tp,tgroup in clgroup.groupby("Train"):
            for d in [msemean,ssimmean,fsimmean,msestd,ssimstd,fsimstd,rounds]: d[tp]=[]
            for sp,sgroup in tgroup.groupby("SplitLayer"):
                lmean=sgroup[["MSE","SSIM","FSIM"]].mean()
                lstd = sgroup[["MSE", "SSIM", "FSIM"]].std().fillna(0)
                msemean[tp].append(lmean["MSE"])
                ssimmean[tp].append(lmean["SSIM"])
                fsimmean[tp].append(lmean["FSIM"])
                msestd[tp].append(lstd["MSE"])
                ssimstd[tp].append(lstd["SSIM"])
                fsimstd[tp].append(lstd["FSIM"])
                rounds[tp].append(sgroup.shape[0])
        meanlist,stdlist,colorlist,labellist=[msemean,ssimmean,fsimmean],[msestd,ssimstd,fsimstd], ["red","green","Blue"], ["MSE","SSIM","FSIM"]
        for mn,sd,ylabel in zip(meanlist,stdlist,labellist):
            c=[colorlist[i] for i in range(len(mn))]
            l=[f"{tp}% {max(rounds[tp])} rounds {min(rounds[tp])} rounds" for tp in trainpercentage if tp in rounds]
            title=f"Dataset {dataset}, Model {model}, Attack {loss},\n Class {cls}"
            outfile=f"{outGraph}{dataset}_{model}_{loss}_{ylabel}_{cls}.png"
            PlotGraphFor(list(mn.values()),list(sd.values()),c,l,ylabel, title, outfile)

def PlotAgrrigationResutls(clgroup,loss,title, outfile,aggrigationtype="max"):
    Path("/".join(outfile.split("/")[:-1])).mkdir(exist_ok=True,parents=True)
    mselist, ssimlist, fsimlist = [], [], []
    for sp, sgroup in clgroup.groupby("SplitLayer"):
        if aggrigationtype=="max":
            mse, ssim, fsim = sgroup[sgroup["Combineloss"] == sgroup["Combineloss"].max()][["MSE", "SSIM", "FSIM"]].values[0]
        if aggrigationtype == "avg":
            mse, ssim, fsim=sgroup.mean()[["MSE", "SSIM", "FSIM"]].values
        mselist.append(mse)
        ssimlist.append(ssim)
        fsimlist.append(fsim)
    PlotGraphFor([mselist, ssimlist, fsimlist], [None, None, None], ["red", "green", "Blue"], ["MSE", "SSIM", "FSIM"],loss.upper(), title, outfile,plotype="Line")


def makeBestResultLossGraph(trainpercentage,dataset,model,loss="mse",imgsize=64,outGraph="Results/Graphs/BestResults/"):
    Path(outGraph).mkdir(exist_ok=True,parents=True)
    losses=pd.read_csv(f"Results/Size_{imgsize}_{imgsize}/CSVFiles/Losses.csv")
    losses=losses.replace("No",0)
    losses["Train"]=losses["Train"].astype(int)
    losses=losses[losses["Dataset"]==dataset]
    losses=losses[losses["Model"]==model]
    losses=losses[losses["LossName"]==loss]
    losses["FSIM"]=losses["FSIM"].fillna(.5)
    del losses["Dataset"],losses["Model"],losses["MAE"],losses["LossName"],losses["SEED"],losses["PredIndex"]
    losses["Combineloss"]=(1-losses["MSE"])+losses["SSIM"]+losses["FSIM"]
    for tp,tplosses in losses.groupby("Train"):
        if tp not in trainpercentage: continue
        for cls,clgroup in tplosses.groupby("Class"):
            title = f"Dataset {dataset}, Model {model}, Attack {loss},\n Class {cls}, {tp}%"
            outfile = f"{outGraph}{dataset}_{model}_{loss}_{cls}_{tp}.png"
            PlotAgrrigationResutls(clgroup,loss,title, outfile,aggrigationtype="max")
        title = f"Dataset {dataset}, Model {model}, Attack {loss}, {tp}%"
        outfile = f"{'/'.join(outGraph.split('/')[:-2])}/AverageResults/{dataset}_{model}_{loss}_{tp}.png"
        PlotAgrrigationResutls(tplosses, loss, title, outfile, aggrigationtype="avg")


def MakeAllMetrices(trainpercentage,losses=["mse"],EnhanceImages=False,convertColor=False,Classes=10):
    metrics_list=("RMSE", "PNSR", "SRE","FSIM","UIQ")
    seeds=np.unique(pd.read_csv("Results/Size_64_64/CSVFiles/Losses.csv")[["SEED"]].values[:,0])

    for index in range(1):
        for model in ["Vgg8Net"]:
            for dataset in ["food"]:
                for loss in losses:
                    # makeAverageLossGraph(trainpercentage,dataset, model,loss)
                    makeBestResultLossGraph(trainpercentage,dataset, model,loss)
                    # for seed in seeds:
                    #     makeInversionGraph(model, dataset, seed,loss=loss, index=index, trainpercentage=tp, EnhanceImages=EnhanceImages,convertColor=convertColor, Classes=Classes, outdir="Results/Graphs/")
                    #     for i in range(len(metrics_list)):
                    #         makeMerircGraph(model, dataset, metrics=metrics_list[i], index=index, trained=trained,Classes=Classes)
                    #     makeInversionGraph(model, dataset,seed, index=index, trained=trained, EnhanceImages=EnhanceImages, convertColor=convertColor,  Classes=10, outdir="Results/Graphs/MetricsGraph/")
    # plt.show()

def MakeLossGraphs(imgsize=64,Train="No",resultFolder="Results/Size_64_64",datalist=None,skipiter=0,n_key_columns=9,updateGraphs=False):
    # lossFiles=[f"Resulcolumnsts/Size_{imgsize}_{imgsize}/CSVFiles/RegenrationLoss.csv",f"ServerData/Results/Size_{imgsize}_{imgsize}/CSVFiles/RegenrationLoss.csv"]
    # losslist,alreadyhave=[],[]
    # for lossFile in lossFiles:
    #     with open(lossFile) as f:
    #         data=[l[:-1].split(",") for l in f.readlines()[1:]]
    #         data=[d for d in data if d[:9] not in alreadyhave]
    #         losslist.extend(data)
    #         alreadyhave.extend([l[:9] for l in losslist])
    Train=[str(T) for T in Train]
    with open(f"Results/Size_{imgsize}_{imgsize}/CSVFiles/RegenrationLoss.csv") as f:
        losslist=[l[:-1].split(",") for l in f.readlines()[1:]]
    losslist=[l for l in losslist if l[0] in Train]
    losslist=sorted(losslist,key=lambda x:",".join(x[:9]))
    for i in tqdm(range(0,len(losslist),2)):
        plt.Figure()
        haveplot=False
        for lossr in losslist[i:i+2]:
            traintype, modelname, dataset,lossname, split_layer, clas, idx, seed, Type=lossr[:9]
            outfilename = f"{resultFolder}/LossGraph/{modelname}_{dataset}_{lossname}_{traintype}/{split_layer}_{clas}_{idx}_{seed}_WithoutTraining_{skipiter}.png"
            if updateGraphs is False and os.path.exists(outfilename):
                plt.close()
                continue
            if datalist is not None and dataset not in datalist: break
            haveplot=True
            plt.plot([float(l) for l in lossr[n_key_columns+skipiter:]], label=lossr[n_key_columns-1])
        if haveplot is False:
            plt.close()
            continue
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        Path("/".join(outfilename.split("/")[:-1])).mkdir(parents=True,exist_ok=True)
        plt.savefig(outfilename)
        plt.close()


def MakeLabelsCSV(folderpath,modelpath,batch_size=64,outfile="Results/Size_64_64/CSVFiles/Pred.csv"):
    columns=["Train","Model","Dataset","LossName","split_layer","Actual class","index","Seed","Pred Class"]
    train_per="0" if folderpath.split("/")[-2]=="WithoutTraining" else folderpath.split("/")[-2].split("_")[1]
    dataset=modelpath.split(".")[0].split("/")[-1].split("_")[1]
    dset=Dataset(folderpath,dataset,outfile,train_percent=train_per)
    if not os.path.exists(outfile):
        with open(outfile, "w") as f:
            f.write(",".join(columns) + "\n")
    if len(dset)==0:
        print("No New Records For",folderpath)
        return
    else:
        print("Processing ",len(dset),"Records for ",folderpath)
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size)
    lables,preds=[],[]
    model=torch.load(modelpath)
    with torch.no_grad():
        for d,l in dloader:
            lables.extend(l)
            pred=model(d.to("cuda"))
            preds.append(torch.argmax(pred,dim=1).cpu().numpy())
    preds=np.concatenate(preds)
    lables=[l+","+str(p)+"\n" for p,l in zip(preds,lables)]
    with open(outfile,"a") as f:
        f.writelines(lables)

def formatPredCsvFiles(predfile="Results/Size_64_64/CSVFiles/Pred.csv",outfile="PredAccuracy.csv"):
    outfile="/".join(predfile.split("/")[:-1])+"/"+outfile
    df=pd.read_csv(predfile)
    dfvalues=[]
    for (t,m,d,l),dfgroup in df.groupby(['Train', 'Model', 'Dataset', 'LossName']):
        for sp,spgroup in dfgroup.groupby("split_layer"):
            actual,pred=spgroup["Actual class"].values,spgroup["Pred Class"].values
            accuracy=accuracy_score(actual,pred)
            dfvalues.append([t,m,d,l,sp,accuracy])
    df=pd.DataFrame(dfvalues,columns=['Train', 'Model', 'Dataset', 'LossName','split_layer','Accuracy'])
    df=df.pivot_table("Accuracy",['Train', 'Model', 'Dataset', 'LossName'],'split_layer').reset_index()
    df.to_csv(outfile,index=None)


def MakePredLabelFileCsv():
    outfile = "Results/Size_64_64/CSVFiles/Pred.csv"
    for trfolder in ["WithoutTraining","WithTraining_30","WithTraining_60"]:
        MakeLabelsCSV("Results/Size_64_64/"+trfolder+"/", "evaluation/Models/Size_64_64/restnet_food.pt",outfile=outfile)
    formatPredCsvFiles(predfile=outfile)


if __name__ == '__main__':
    # CombineLosses()
    MakeAllMetrices([0,30,60],convertColor=True,losses=["mse","fsim","ssim"])
    # for sk in [0,5,50,100,500]:
    #     MakeLossGraphs(Train=[30],skipiter=sk,datalist=["food"])
    # MakePredLabelFileCsv()


