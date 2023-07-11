import os,sys,argparse,torch
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.io import read_image
from evaluation.EvaluationMetrics import Evaluation

import unsplit.attacks as unsplit
from DataSet import getDataAndModels,datasetslist,modelList
from unsplit.util import *

def makeBaseLossFile(lossFile="Losses.csv",rloss="RegenrationLoss.csv"):
     if not os.path.exists(lossFile):
         Path(lossFile.replace("Losses.csv","")).mkdir(parents=True, exist_ok=True)
         with open(lossFile, "w") as f:
             f.write(f'Train,Model,Dataset,LossName,SplitLayer,Class,Index,SEED,MSE,MAE,SSIM,PredIndex\n')
     if not os.path.exists(rloss):
         Path(rloss.replace("RegenrationLoss.csv","")).mkdir(parents=True, exist_ok=True)
         with open(rloss, "w") as f:
            f.write(f'Train,Model,Dataset,LossName,SplitLayer,Class,Index,SEED,Type,Loss\n')




def Training(server,client,split_layer,trainset,testset,dataset,modelname,modelpath="Models/",epochs = 100,trainpercent=100):
    # -- TRAIN MODELS --
    epochs=int(epochs*trainpercent/100)
    print('Training models for ',trainpercent,"% that is ",epochs)
    Path(modelpath).mkdir(exist_ok=True,parents=True)
    serverpath=modelpath+"/"+modelname+"_"+dataset+"_Server_"+str(epochs)+"_"+str(split_layer)+".pt"
    clientpath=modelpath+"/"+modelname+"_"+dataset+"_Client_"+str(epochs)+"_"+str(split_layer)+".pt"
    if os.path.exists(serverpath) and os.path.exists(clientpath):
        print("Found Trained Model Loading that")
        server=torch.load(serverpath)
        client=torch.load(clientpath)
        return server,client

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server,client=server.to(device),client.to(device)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)
    # testloader = torch.utils.data.DataLoader(testset, shuffle=True)

    client_opt = torch.optim.Adam(client.parameters(), lr=0.001, amsgrad=True)
    server_opt = torch.optim.Adam(server.parameters(), lr=0.001, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print("Training for epoch",epoch)
        running_loss = 0
        for images, labels in tqdm(trainloader):
            images,labels=images.to(device),labels.to(device)
            client_opt.zero_grad()
            server_opt.zero_grad()
            pred = server(client(images, end=split_layer), start=split_layer+1)
            loss = criterion(pred, labels)
            loss.backward()
            running_loss += loss
            server_opt.step()
            client_opt.step()
        else:
            print(f'Epoch: {epoch} Loss: {running_loss / len(trainloader)} Acc: {get_test_score(client, server, testset,device, split=split_layer)}')
    print('Training Done.')
    server,client=server.cpu(),client.cpu()
    torch.save(server, serverpath)
    torch.save(client, clientpath)
    return server.cpu(), client.cpu()

def saveResults(instanceinfo,lossname,loss,result,target,resultFolder,traintype,dataset,modelname,split_layer,clas,idx,seed,inputloss,clonemodellos,device,evalobject):
    # create results directory if doesn't exist.
    Path(resultFolder).mkdir(parents=True, exist_ok=True)
    pred,  SSIM = evalobject.calculateScore(result, target)
    # save_image(result, f'Results/{traintype}/{modelname}_{dataset}_{split_layer}_{clas}_CCH_{idx}.png')
    print(f'{instanceinfo},{loss},{torch.nn.L1Loss()(result, target.to(device))},{SSIM},{pred}\n')
    save_image(result, f'{resultFolder}{traintype}/{modelname}_{dataset}_{lossname}_{split_layer}_{clas}_{idx}_{seed}.png')
    with open(f"{resultFolder}CSVFiles/Losses.csv", "a") as f:
        f.write(f'{instanceinfo},{loss},{torch.nn.L1Loss()(result, target.to(device))},{SSIM},{pred}\n')
    with open(f"{resultFolder}CSVFiles/RegenrationLoss.csv", "a") as f:
        f.write(f'{instanceinfo},INPUT,{",".join([str(inp) for inp in inputloss])}\n')
        f.write(f'{instanceinfo},Model,{",".join([str(cln) for cln in clonemodellos])}\n')
    plt.Figure()
    plt.plot(inputloss,label="inputloss")
    plt.plot(clonemodellos,label="clonemodellos")
    plt.xlabel("Epoches")
    plt.ylabel("Labels")
    plt.legend()
    os.makedirs(f"{resultFolder}/LossGraph",exist_ok=True)
    plt.savefig(f"{resultFolder}/LossGraph/{modelname}_{dataset}_{lossname}_{split_layer}_{clas}_{idx}_{seed}_{traintype}.png")
    plt.close()


def getInversionTargets(testset,dataset):
    if isinstance(testset, Dataset):
        inversion_targets = [get_examples_by_class(testset, c, count=2) for c in range(10)]
    else:
        inversion_targets = [[read_image(testset + f).float() for f in sorted(os.listdir(testset)) if f.startswith(dataset+"_"+str(i))] for i in range(10)]
        inversion_targets = [[inv/torch.max(inv) for inv in invtarget] for invtarget in inversion_targets]
        if "mnist" in dataset:
            inversion_targets=[[torch.mean(inv,dim=1,keepdim=True) for inv in invtarget] for invtarget in inversion_targets]
    inversion_targets = [[torch.unsqueeze(inv, dim=0) for inv in invtarget] for invtarget in inversion_targets]
    return inversion_targets

def InversionAttack(args,client,clone,split_layer,inversion_targets,dataset,seed,device,resultFolder="Results/"):
    # -- MODEL INVERSION & STEALING --
    try:
        print('Starting model inversion & stealing attack...')
        traintype =f"WithTraining_{args.trainPercentage}" if args.train else "WithoutTraining"
        resultFolder=f"{resultFolder}Size_{args.imagesize}_{args.imagesize}/"
        Path(f"{resultFolder}{traintype}").mkdir(exist_ok=True,parents=True)
        # load one example per class from the test set
        mse = torch.nn.MSELoss()
        eval=Evaluation(dataset)
        makeBaseLossFile(f"{resultFolder}CSVFiles/Losses.csv",f"{resultFolder}CSVFiles/RegenrationLoss.csv")
        with open(f"{resultFolder}CSVFiles/Losses.csv") as f:
            processedData=[",".join(d.split(",")[:8]) for d in f.readlines()[1:]]
        results, losses = [], []
        for clas, target_list in enumerate(inversion_targets):
            if clas not in args.classes: continue
            for idx, target in enumerate(target_list):
                if idx not in args.indexes: continue
                instanceinfo=f'{args.trainPercentage if args.train else "No"},{args.model},{dataset},{args.loss},{split_layer},{clas},{idx},{seed}'
                if not args.allowrewrite and instanceinfo in processedData:
                    print(f"combination {instanceinfo} already exists")
                    continue
                print(f"Doing Inversion for model {args.model}, dataset {dataset}, Loss {args.loss}, split {split_layer}, class {clas}, Index {idx}, Seed  {seed}, ImageSize {args.imagesize} ")
                # obtain client output
                client_out = client(target, end=split_layer)
                # perform the attack
                result,inputloss,clonemodellos = unsplit.model_inversion_stealing(clone, split_layer, client_out, target.size(),lossname=args.loss,
                                                          main_iters=args.inversionmainteration, input_iters=100, model_iters=100,device=device)
                # save result
                if dataset in ['cifar',"food","flower"]:
                    result = normalize(result)
                results.append(result)
                loss = mse(result, target.to(device))
                losses.append(loss)
                saveResults(instanceinfo,args.loss, loss, result, target, resultFolder, traintype, dataset, args.model,
                            split_layer, clas, idx, seed, inputloss, clonemodellos, device,eval)
                processedData.append(instanceinfo)

        # print(f'Average MSE: {sum(losses) / len(losses)}')
        print(f'Results saved to the results directory.')
    except Exception as e:
        print("Exception",e)



def getArguments():
    parser = argparse.ArgumentParser(
                        prog='Inversion Attack',
                        description='It Do the Inverse Untack on midele  output')

    parser.add_argument('-d', '--dataset',default=["food"],nargs="+",choices=datasetslist,help="Dataset to use")
    parser.add_argument('-m', '--model',default="Vgg8Net",choices=modelList,help="Model to use")
    parser.add_argument('-c', '--cuda',action="store_true",help="Whether to use cuda?")
    parser.add_argument('-arw', '--allowrewrite',action="store_true",help="Whether You want to allow rewrite the exsiting results if exist?")
    parser.add_argument('-l', '--splitlayer',default=[1,2,3,4,5,6],nargs="+",type=int,choices=range(1,7),help="Split Layer")
    parser.add_argument('-cl', '--classes',default=range(10),nargs="+",type=int,choices=range(10),help="for which class label want to run")
    parser.add_argument('-s', '--seed',default=[random.randint(0,50000)], nargs="+",type=int,help="Seed to run it with same ")
    parser.add_argument('-i', '--indexes', default=[0], nargs="+", type=int, choices=range(2),help="for which index want to run")
    parser.add_argument('-t', '--train',action="store_true",help="Whether To Train")
    parser.add_argument('-im', '--imagesize',default=64,help="Imagesize to Train",choices=[32,64,128])
    parser.add_argument('-tp', '--trainPercentage',default=100,type=int,help="What Percentage Train if train is selected")
    parser.add_argument('-ls', '--loss',default="mse",choices=["mse","ssim"],help="Choose loss to Use")
    parser.add_argument('-rs', '--results',default="Results",help="Direcotry to save Results")
    parser.add_argument('-tri', '--trainiteration',default=34,type=int,help="Number of iterations to Train")
    parser.add_argument('-imi', '--inversionmainteration',default=1000,type=int,help="Number of Inversion Main iterations")
    args = parser.parse_args()
    # seed used 84554 12564 35468 74852 40864 42951 45769 47821 58964 98723
    print(args)
    device=torch.device("cuda" if args.cuda else "cpu")
    if args.results=="ResultsDel": print("** Result will be saved at ResultsDel ***")
    # dataset,split_layers,device,allowedIndex,train="cifar",[1,2,3,4,5,6],torch.device("cpu"),None,False
    return args,device

def RunCode():
    args,device=getArguments()
    for dataset in args.dataset:
        for seed in args.seed:
            set_random_seed(seed)
            print("Working On Seed", seed)
            trainset, testset, client, server, clone = getDataAndModels(dataset, args.model,datasize=(args.imagesize,args.imagesize))
            inversion_targets = getInversionTargets(testset,dataset)
            for split_layer in args.splitlayer:
                if args.train: server, client = Training(server, client, split_layer, trainset, testset,dataset, args.model,modelpath=f"Models/Size_{args.imagesize}_{args.imagesize}", epochs=args.trainiteration,trainpercent=args.trainPercentage)
                InversionAttack(args,client,clone,split_layer,inversion_targets,dataset,seed,device=device,resultFolder=args.results+"/")
                # if isinstance(testset, Dataset): print(f'Clone test score: {get_test_score(client, clone.cpu(), testset,device, split=split_layer)}%')

if __name__ == '__main__':
    # dataset, model="food","CifarNet"
    # trainset, testset, client, server, clone = getDataAndModels(dataset, model)
    # inversion_targets = getInversionTargets(testset, dataset)
    # for d in inversion_targets:
    #     print(torch.max(d[0]),torch.min(d[0]))

    RunCode()

