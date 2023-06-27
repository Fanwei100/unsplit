import os,sys
from torchviz import make_dot
from matplotlib import pyplot as plt
import torch,numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import  models
sys.path.append("../")
from DataSet import getDataAndModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotaccuracy(trainaccuracy,testacc,dataset,modelname,datasize=(32,32),outfolder="Graph/",graphname=""):
    if len(graphname)>0: graphname="_"+graphname
    plt.figure()
    plt.plot(trainaccuracy,label="Train")
    plt.plot(testacc,label="Test")
    plt.title(f"{dataset} accuracy Graph for {datasize[0]} X {datasize[1]}")
    plt.legend()
    os.makedirs(outfolder,exist_ok=True)
    plt.savefig(f"{outfolder}{modelname}_{dataset}_{datasize[0]}_{datasize[1]}{graphname}.png")
    plt.close()

def saveResults(model,trainaccuracylist,testacclist,datasize,modelname,dataset,epoch):
    os.makedirs("Models",exist_ok=True)
    torch.save(model,f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_{epoch}_{trainaccuracylist[-1]:.3f}_{testacclist[-1]:.3f}.pt")
    plotaccuracy(trainaccuracylist, testacclist, dataset, modelname=modelname, datasize=datasize)
    os.makedirs("Results",exist_ok=True)
    with open(f"Results/{modelname}_{dataset}_{datasize[0]}_{datasize[1]}.csv","w") as f:
        f.write("Epoch,"+",".join([str(t) for t in range(len(trainaccuracylist))])+"\n")
        f.write("Train,"+",".join([str(t) for t in trainaccuracylist])+"\n")
        f.write("Test,"+",".join([str(t) for t in testacclist]))


def Training(model,trainset,testset,dataset,opt=None,stepoch=0,epochs = 100,datasize=(32,32),batch_size=512):
    # -- TRAIN MODELS --
    print('Training models...',dataset)
    os.makedirs("Models",exist_ok=True)
    os.makedirs(f'Models/Size_{datasize[0]}_{datasize[1]}',exist_ok=True)

    model=model.to(device)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size)

    if opt is None: opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    modelname=model.getName()
    try: del model.getName
    except: pass
    trainaccuracylist,testacclist=[],[]
    for epoch in range(stepoch,epochs):
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
        if (len(testacclist)<50 and epoch%10==9) or (len(testacclist)>50 and testacclist[0]==max(testacclist)):
            saveResults(model, trainaccuracylist, testacclist, datasize, modelname, dataset, epoch)
    saveResults(model, trainaccuracylist, testacclist, datasize, modelname, dataset, epoch)
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
        if epoch%10==9:
            torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Client_{epoch}.pt")
            torch.save(serverModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Server_{epoch}.pt")
            plotaccuracy(trainaccuracylist, testacclist,dataset,modelname=modelname,datasize=datasize,graphname="Ep"+str(epoch))
    torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Client.pt")
    torch.save(clientModel, f"Models/Size_{datasize[0]}_{datasize[1]}/{modelname}_{dataset}_Server.pt")
    plotaccuracy(trainaccuracylist, testacclist,dataset,modelname=modelname,datasize=datasize)
    print(trainaccuracylist)
    print(testacclist)
    print("Max accuracies",np.max(trainaccuracylist),np.max(testacclist))
    print('Training Done.')

def checkModel(model):
    x=torch.randn((10,3,64,64)).to("cuda")
    del model.layer4
    names="model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.avgpool,model.fc".split(",")
    for i,l in enumerate([model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.avgpool,model.fc]):
        x=l(x)
        print(names[i],x.shape)
    print(model)


def buildModel(name="restnet18",pretrained=True,num_classes=10):
    if "restnet" in name:
        if name=="customrestnet":
            model=torch.load("Models/Size_256_256/restnet18_food_9.pt")
            del model.avgpool,model.layer4
            # checkModel(model)
        if name=="restnet18":
            model = models.resnet18(pretrained=pretrained)
        elif name == "restnet34":
            model = models.resnet34(pretrained=pretrained)
        elif name=="restnet50":
            model = models.resnet50(pretrained=pretrained)
        elif name=="restnet101":
            model = models.resnet101(pretrained=pretrained)
        model.requires_grad=False
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        model.fc = nn.Linear(num_ftrs, num_classes)
        opt=torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    elif "vgg" in name:
        if name=="vgg11":
            model = models.vgg11(pretrained=True)
        if name=="vgg19":
            model = models.vgg19(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        # Here the size of each output sample is set to 2.
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    elif "efficientnet" in name:
        if name == "efficientnet_b0":model = models.efficientnet_b0(pretrained=True)
        elif name == "efficientnet_b1":model = models.efficientnet_b1(pretrained=True)
        elif name == "efficientnet_b2":model=torch.load("Models/Size_64_64/efficientnet_b2_food_79_0.997_0.729.pt") #model = models.efficientnet_b2(pretrained=True)
        elif name == "efficientnet_b3":model = models.efficientnet_b3(pretrained=True)
        elif name == "efficientnet_b4":model = models.efficientnet_b4(pretrained=True)
        elif name == "efficientnet_b5":model = models.efficientnet_b5(pretrained=True)
        elif name == "efficientnet_b6":model = models.efficientnet_b6(pretrained=True)
        elif name == "efficientnet_v2_s":model = models.efficientnet_v2_s(pretrained=True)
        elif name == "efficientnet_v2_m":model = models.efficientnet_v2_m(pretrained=True)
        # Here the size of each output sample is set to 2.
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    elif "inception" in name:
        if name == "inception_v3":model = models.inception_v3(pretrained=True)
        # Here the size of each output sample is set to 2.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)


    def getName():
        return name
    model.getName=getName
    return model,opt

def makePlot(model,datasize,modelname,outfolder="Graph/"):
    make_dot(model(torch.randn((1,3,*datasize))).mean(), params=dict(model.named_parameters())).render(outfolder+modelname, format="png")

def TrainLargeModel():
    dataset,datasize,epochs,batch_size,modelname,opt="food",(256,256),80,64,"restnet18",None
    print("Training with Size",datasize)
    trainset, testset, clientModel, serverModel, _ = getDataAndModels(dataset, model="Vgg8Net", datasize=datasize)
    clientModel, opt = buildModel(modelname, pretrained=True)
    Training(clientModel, trainset, testset, dataset, opt=opt, epochs=epochs, datasize=datasize, batch_size=batch_size)


def TrainModel(splitpoint=None):
    datasize,epochs,batch_size,modelname,opt=(64,64),150,132,"efficientnet_b6",None
    datasetslist=["food","flower"]
    for dataset in datasetslist[:1]:
        trainset,testset,clientModel, serverModel, _=getDataAndModels(dataset,model="Vgg8Net",datasize=datasize)
        clientModel,opt = buildModel(modelname,pretrained=True)
        if splitpoint:
            TrainingSplitModel(clientModel, serverModel, splitpoint, trainset, testset, dataset, epochs=epochs,datasize=datasize, batch_size=512)
        else:
            Training(clientModel,trainset,testset,dataset,opt=opt,epochs =epochs,datasize=datasize,batch_size=batch_size)
            # Training(clientModel,trainset,testset,dataset,opt=opt,stepoch=80,epochs =epochs,datasize=datasize,batch_size=batch_size)

    plt.show()

def TrainModelOnDiffSize():
    dataset,datasize,epochs,batch_size,modelname,opt="food",(64,64),150,132,"efficientnet_v2_s",None
    batch_base=1
    datasize_list=[(128,128), (256,256), (512,512), (1024,1024)]
    batch_sizes=[bs*batch_base for bs in [90,20,4,1]]
    for datasize,batch_size in zip(datasize_list,batch_sizes):
        print(f"Training for {batch_size} batch size and {datasize} datasize {modelname} model")
        trainset,testset,clientModel, serverModel, _=getDataAndModels(dataset,model="Vgg8Net",datasize=datasize)
        clientModel,opt = buildModel(modelname,pretrained=True)
        Training(clientModel,trainset,testset,dataset,opt=opt,epochs =epochs,datasize=datasize,batch_size=batch_size)


TrainModelOnDiffSize()
# TrainLargeModel()
# TrainModel()
# dataset=sys.argv[1]
# trainset,testset,model, _, _=getDataAndModels(dataset,datasize=datasize)
# model= buildRestNet()
# Training(model,trainset,testset,dataset,epochs = int(sys.argv[2]),datasize=datasize,batch_size=32)

