import os

import numpy as np
from torchvision import transforms, datasets
from unsplit.models import *

datapath="/media/parmpal/Workspace/DataSet/TorchData/"
if not os.path.exists(datapath): datapath="data/"

datasetslist=["mnist","f_mnist","cifar","food","flower"]
modelList=["MnistNet","CifarNet","Vgg8Net","Restnet"]
datasizes={"mnist":(28,28),"f_mnist":(28,28),"cifar":(32,32),"food":(32,32),"flower":(32,32)}

def FilterDataWithLables(trainset,lables):
    indexies=[i for i,l in enumerate(trainset._labels) if l in lables ]
    trainset._image_files=[trainset._image_files[i] for i in indexies]
    trainset._labels=[trainset._labels[i] for i in indexies]
    return trainset

def getDataAndModels(dataset,model=None,datasize=None):
    # load datasets and initialize client, server, and clone models
    if datasize is None: datasize=datasizes[dataset]
    if model is None: # set Default Model For Data
        model={"mnist":"MnistNet","f_mnist":"MnistNet","cifar":"CifarNet","food":"Vgg8Net","flower":"Vgg8Net"}[dataset]
    assert dataset in datasetslist," Only "+str(datasetslist)+" supported"
    mytransforms = transforms.Compose([
                 transforms.Resize(datasize),
                 transforms.ToTensor(),
    ])
    if dataset == 'mnist':
        trainset = datasets.MNIST(datapath+'mnist', download=True, train=True, transform=mytransforms)
        testset = datasets.MNIST(datapath+'mnist', download=True, train=False, transform=mytransforms)
    elif dataset == 'f_mnist':
        trainset = datasets.FashionMNIST(datapath+'f_mnist', download=True, train=True, transform=mytransforms)
        testset = datasets.FashionMNIST(datapath+'f_mnist', download=True, train=False, transform=mytransforms)
    elif dataset == 'cifar':
        trainset = datasets.CIFAR10(datapath+'cifar', download=True, train=True, transform=mytransforms)
        testset = datasets.CIFAR10(datapath+'cifar', download=True, train=False, transform=mytransforms)
    elif dataset == 'food':
        trainset = datasets.Food101(datapath + 'food101', download=True, split="train", transform=mytransforms)
        testset = datasets.Food101(datapath + 'food101', download=True, split="test", transform=mytransforms)
        trainset, testset = FilterDataWithLables(trainset, list(range(10))), FilterDataWithLables(testset,list(range(10)))
    elif dataset == 'flower':
        trainset = datasets.Flowers102(datapath+'flower', download=True, split="train", transform=mytransforms)
        testset  = datasets.Flowers102(datapath+'flower', download=True, split="test", transform=mytransforms)
        trainset, testset = FilterDataWithLables(trainset, list(range(10))), FilterDataWithLables(testset, list(range(10)))
    if model=="MnistNet":
        client, server, clone = MnistNet(), MnistNet(), MnistNet()
    elif model=="CifarNet":
        client, server, clone = CifarNet(), CifarNet(), CifarNet()
    elif model=="Vgg8Net":
        client, server, clone = VGG8(), VGG8(), VGG8()
    elif model=="Restnet":
        client, server, clone = ResNet18(), ResNet18(), ResNet18()

    return trainset,testset,client, server, clone

