import os
import sys
from matplotlib import pyplot as plt
import torch,numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import  models
sys.path.append("../")
from DataSet import getDataAndModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotaccuracy(trainaccuracy,testacc,dataset,modelname,datasize=(32,32),outfolder="Graph/",graphname=""):
    plt.figure()
    plt.plot(trainaccuracy,label="Train")
    plt.plot(testacc,label="Test")
    plt.title(f"{dataset} accuracy Graph for {datasize[0]} X {datasize[1]}")
    plt.legend()
    os.makedirs(outfolder,exist_ok=True)
    plt.savefig(f"{outfolder}{modelname}_{dataset}_{datasize[0]}_{datasize[1]}_{graphname}.png")
    plt.close()


def Training(model,trainset,testset,dataset,epochs = 100,datasize=(32,32),batch_size=512):
    # -- TRAIN MODELS --
    print('Training models...',dataset)
    os.makedirs("Models",exist_ok=True)
    os.makedirs(f'Models/Size_{datasize[0]}_{datasize[1]}',exist_ok=True)

    model=model.to(device)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    modelname=model.getName()
    try: del model.getName
    except: pass
    trainaccuracylist,testacclist=[],[]
    for epoch in range(epochs):
        print("------------Epoch "+str(epoch)+"---------------")
        labelslist, predlist = [], []
        for images, labels in tqdm(trainloader):
            images,labels=images.to(device),labels.to(device)
            opt.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()
            labelslist.append(labels)
            predlist.append(torch.argmax(pred, dim=1))
        labels, preds = torch.hstack(labelslist).cpu().numpy(), torch.hstack(predlist).cpu().numpy()
        trainaccuracylist.append(np.sum(labels==preds)/labels.shape[0])
        with torch.no_grad():
            labelslist,predlist=[],[]
            for images, labels in tqdm(testloader):
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                labelslist.append(labels)
                predlist.append(torch.argmax(pred,dim=1))
            labels, preds = torch.hstack(labelslist).cpu().numpy(), torch.hstack(predlist).cpu().numpy()
            testacclist.append(np.sum(labels == preds) / labels.shape[0])
        print(f"Epoch {epoch} TrainAcc {trainaccuracylist[-1]} Testacc {testacclist[-1]}")
        if epoch%5==4: torch.save(model, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_{epoch}.pt")
    torch.save(model, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}.pt")
    plotaccuracy(trainaccuracylist, testacclist,dataset,modelname=modelname,datasize=datasize)
    print(trainaccuracylist)
    print(testacclist)
    print("Max accuracies",np.max(trainaccuracylist),np.max(testacclist))
    print('Training Done.')


def TrainingSplitModel(clientModel,serverModel,splitpoint,trainset,testset,dataset,epochs = 100,datasize=(32,32),batch_size=512):
    # -- TRAIN MODELS --
    print('Training models With Split Learning at point',splitpoint,"With data",dataset)
    os.makedirs("Models",exist_ok=True)
    os.makedirs(f'Models/Size_{datasize[0]}_{datasize[1]}',exist_ok=True)

    clientModel,serverModel=clientModel.to(device),serverModel.to(device)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size)

    opt = torch.optim.Adam(list(clientModel.parameters()) + list(serverModel.parameters()), lr=0.001, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    modelname=clientModel.getName()
    trainaccuracylist,testacclist=[],[]
    for epoch in range(epochs):
        print("------------Epoch "+str(epoch)+"---------------")
        labelslist, predlist = [], []
        for images, labels in tqdm(trainloader):
            images,labels=images.to(device),labels.to(device)
            opt.zero_grad()
            pred = serverModel(clientModel(images,end=splitpoint),start=splitpoint)
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()
            labelslist.append(labels)
            predlist.append(torch.argmax(pred, dim=1))
        labels, preds = torch.hstack(labelslist).cpu().numpy(), torch.hstack(predlist).cpu().numpy()
        trainaccuracylist.append(np.sum(labels==preds)/labels.shape[0])
        with torch.no_grad():
            labelslist,predlist=[],[]
            for images, labels in tqdm(testloader):
                images, labels = images.to(device), labels.to(device)
                pred = serverModel(clientModel(images, end=splitpoint), start=splitpoint)
                labelslist.append(labels)
                predlist.append(torch.argmax(pred,dim=1))
            labels, preds = torch.hstack(labelslist).cpu().numpy(), torch.hstack(predlist).cpu().numpy()
            testacclist.append(np.sum(labels == preds) / labels.shape[0])
        print(f"Epoch {epoch} TrainAcc {trainaccuracylist[-1]} Testacc {testacclist[-1]}")
        if epoch%5==4:
            torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Client_{epoch}.pt")
            torch.save(serverModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Server_{epoch}.pt")
        if epoch%10==9:
            plotaccuracy(trainaccuracylist, testacclist,dataset,modelname=modelname,datasize=datasize,graphname="Ep"+str(epoch))
    torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Client.pt")
    torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Server.pt")
    plotaccuracy(trainaccuracylist, testacclist,dataset,modelname=modelname,datasize=datasize)
    print(trainaccuracylist)
    print(testacclist)
    print("Max accuracies",np.max(trainaccuracylist),np.max(testacclist))
    print('Training Done.')


def buildModel(name="restnet",pretrained=True,num_classes=10):
    if name=="restnet":
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif name=="vgg11":
        model = models.vgg11(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        # Here the size of each output sample is set to 2.
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def getName():
        return name
    model.getName=getName
    return model



def TrainModel(splitpoint=None):
    datasize,epochs,batch_size,modelname=(64,64),80,1024,"restnet"

    datasetslist=["food","flower"]
    for dataset in datasetslist[:1]:
        trainset,testset,clientModel, serverModel, _=getDataAndModels(dataset,model="Vgg8Net",datasize=datasize)
        # model = buildModel(modelname,pretrained=True)
        if splitpoint:
            TrainingSplitModel(clientModel, serverModel, splitpoint, trainset, testset, dataset, epochs=epochs,datasize=datasize, batch_size=512)
        else:
            Training(clientModel,trainset,testset,dataset,epochs =epochs,datasize=datasize,batch_size=batch_size)

    plt.show()

TrainModel(splitpoint=8)
# dataset=sys.argv[1]
# trainset,testset,model, _, _=getDataAndModels(dataset,datasize=datasize)
# model= buildRestNet()
# Training(model,trainset,testset,dataset,epochs = int(sys.argv[2]),datasize=datasize,batch_size=32)