import os.path
import shutil
from pathlib import Path

import pandas as pd

from DataSet import getDataAndModels
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# def makeGraphForLabel(dataset,n=5):
#     images=[[img for img,l in dataset if l==i][:n] for i in range(10)]
#     images=[np.rollaxis(img.numpy(),0,3) for imgs in images for img in imgs]
#     fig = plt.figure(figsize=(20., 15.))
#     grids = ImageGrid(fig, 111,  # similar to subplot(111)
#                      nrows_ncols=(10, n),  # creates 2x2 grid of axes
#                      axes_pad=0.1,  # pad between axes in inch.
#                      )
#
#     for ax,img in zip(grids,images):
#         ax.imshow(img)
#     plt.show()

# dataset="flower"
# trainset,testset,client, server, clone=getDataAndModels(dataset)
#
# makeGraphForLabel(trainset)
# import cv2,numpy as np
# path="/media/parmpal/Data/python/upwork/Yvette_Zoran/FederatadLearning/SplitLearning/unsplit/ServerData/Results/WithoutTraining/CifarNet_flower_2_8_0.png"
# print(np.max(cv2.imread(path)))

# with open("Losses.csv") as f:
#     data1=set([",".join(d.split(",")[:6]) for d in f.readlines()[1:]])
# with open("ServerData/Losses.csv") as f:
#     data2=set([",".join(d.split(",")[:6]) for d in f.readlines()[1:]])
#
# print(len(data1),len(data2),len(data1.union(data2)),)
# for d in data1.intersection(data2):
#     print(d)
# def copyImagesAfterCorrectName():
#     data=pd.read_csv("/media/parmpal/Data/python/upwork/Yvette_Zoran/FederatadLearning/SplitLearning/unsplit/ServerData/Results/Size_64_64/CSVFiles/Losses.csv")
#     data=data[data["LossName"]=="ssim"]
#     sourceFolder="/media/parmpal/Data/python/upwork/Yvette_Zoran/FederatadLearning/SplitLearning/unsplit/ServerData/Results/Size_64_64/WrongImageName/"
#     destFolder="/media/parmpal/Data/python/upwork/Yvette_Zoran/FederatadLearning/SplitLearning/unsplit/ServerData/Results/Size_64_64/WithoutTraining/"
#     for i,d in data.iterrows():
#         spath=f"{sourceFolder}{d['Model']}_{d['Dataset']}_{d['SplitLayer']}_{d['Class']}_{d['Index']}_{d['SEED']}.png"
#         dpath=f"{destFolder}{d['Model']}_{d['Dataset']}_ssim_{d['SplitLayer']}_{d['Class']}_{d['Index']}_{d['SEED']}.png"
#         print(os.path.exists(spath),os.path.exists(dpath))
#         if not os.path.exists(dpath):
#             shutil.copy(spath,dpath)

def MergeData(fileList,n_columns=8):
    print("******************** Merging",fileList[0].split("/")[-1],"Files *********************")
    columns,values=None,[]
    for fl in fileList:
        with open(fl) as f:
            data=f.readlines()
        if data[0].split(",")[0]=="Train":
            columns=data[0]
            values.extend(data[1:])
        else:
            values.extend(data)
        print(fl,"have",len(data))
    print("Have columns",columns,end="")
    print("Before Duplicates",len(values))
    # remove Duplicates
    values=list({",".join(val.split(",")[:n_columns]):val for val in values}.values())
    print("After remove the Duplicates",len(values))
    values.insert(0,columns)
    with open(fileList[0],"w") as f:
        f.writelines(values)
    print("******************** Merging Finished *********************")



def copyFiles(sources,dest):
    print("Coping Files......")
    print("From",sources)
    print("To",dest)
    Path(dest).mkdir(exist_ok=True,parents=True)
    for source in sources:
        if not os.path.exists(source): continue
        dfiles = os.listdir(dest)
        sfiles=[f for f in os.listdir(source) if f not in dfiles]
        for s in sfiles:
            shutil.copy(source+s,dest)
        print("Copied ",len(sfiles)," Files from ",source)
    print("Copy Finished......")



# copyImagesAfterCorrectName()
def copyToLocalFiles():
    resultBases=["","ServerData/","ServerData2/","Server_Results 6_26/","Server_SL_Results/"]
    MergeData([b+"Results/Size_64_64/CSVFiles/Losses.csv" for b in resultBases],n_columns=8)
    MergeData([b+"Results/Size_64_64/CSVFiles/RegenrationLoss.csv" for b in resultBases],n_columns=9)
    for folder in ["WithoutTraining","WithTraining_30","WithTraining_60"]:
        copyFiles([b+"Results/Size_64_64/"+folder+"/" for b in resultBases], "Results/Size_64_64/"+folder+"/")

copyToLocalFiles()

# def AddMse(basepath="Results/Size_64_64/WithoutTraining/"):
#     for f in os.listdir(basepath):
#         d=f
#         if len(f.split("_"))==6:
#             d=d.split("_")
#             d.insert(2,"mse")
#             d="_".join(d)
#             shutil.move(basepath+f,basepath+d)
#         print(len(f.split("_")),f,d)
#
# AddMse()
