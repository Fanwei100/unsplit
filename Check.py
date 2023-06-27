# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
import os

import pandas as pd
import tqdm
from image_similarity_measures.quality_metrics import rmse, psnr,sre,ssim,fsim,issm,uiq,sam
#
# # Calculate Euclidean distance between two points
# point1 = np.random.random((10,32,32,5))
# point2 = np.random.random((10,32,32,5))
#
# # print(fsim(point1,point2))
# print(ssim(point1,point2))

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *


def eval_step(engine, batch):
    return batch
default_evaluator = Engine(eval_step)

def calualrteSSIM(preds,target):
    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')
    state = default_evaluator.run([[preds, target]])
    return state.metrics['ssim']



import torch.nn.functional as F
def ssim_loss(img1, img2, window_size=11, size_average=True):
    # [(2 * μx * μy + C1) * (2 * σxy + C2)] / [(μx ^ 2 + μy ^ 2 + C1) * (σx ^ 2 + σy ^ 2 + C2)]
    # Compute SSIM
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    mu1 = F.avg_pool2d(img1, window_size, stride=1)
    mu2 = F.avg_pool2d(img2, window_size, stride=1)

    sigma1 = F.avg_pool2d(img1 ** 2, window_size, stride=1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, window_size, stride=1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1) - mu1 * mu2
    SSIM = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    if size_average:
        return 1 - SSIM.mean()
    else:
        return 1 - SSIM.mean(1).mean(1).mean(1)


def fsim_loss(img1, img2, alpha=0.4, beta=0.4, gamma=0.2):
    # Compute FSIM
    # (2 * μx * μy + α) * (2 * σxy + β) / (μx ^ 2 + μy ^ 2 + α) * (σx ^ 2 + σy ^ 2 + β)

    mu1 = F.avg_pool2d(img1, 3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, stride=1, padding=1)

    sigma1 = F.avg_pool2d(img1 ** 2, 3, stride=1, padding=1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 3, stride=1, padding=1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, stride=1, padding=1) - mu1 * mu2

    numerator = (2 * sigma12 + alpha) * (2 * mu1 * mu2 + beta)
    denominator = (sigma1 + sigma2 + alpha) * (mu1 ** 2 + mu2 ** 2 + beta)
    print(img1.shape,img2.shape)
    print(mu1.shape,mu2.shape,sigma1.shape,sigma2.shape,sigma12.shape)
    print(numerator.shape,denominator.shape)
    fsim_map = numerator / denominator

    return (1 - fsim_map.mean()) ** gamma

# t1,t2=torch.randn((10,7,25,25)),torch.randn((10,7,25,25))
# # print(calualrteSSIM(t1,t2))
# # print(ssim_loss(t1,t2))
# # print(ssim(t1.numpy(),t2.numpy()))
# print(fsim_loss(t1,t2))
# # Metrics cn be used  RMSE,MSE,MAE,R-Sequered we can also use FSIM,SSIM,SAM,UIQ,SRE,PNSR,ISIM but some of these give weird output so that need to check.
# epochs,trainpercent=33,30
# epochs,trainpercent=10,30
# print(epochs,trainpercent)
# print(epochs/trainpercent*100)
#
# # x/trainpercent*100=10*trainpercent/100
# epochs,trainpercent=34,30
# print(epochs,trainpercent)
# print(epochs*trainpercent/100)


def DataAnalysis():
    from torchvision import transforms, datasets
    datapath="/media/parmpal/Workspace/DataSet/TorchData/"
    trainset = datasets.Food101(datapath + 'food101', download=True, split="train", transform=None)
    clss=list(range(10))

    datalist={}
    for t,l in tqdm.tqdm(trainset):
        if l in clss:
            if l not in datalist:
                datalist[l]={}
            sz = "x".join([str(l) for l in t.size])
            if sz in datalist[l]:
                datalist[l][sz]+=1
            else:
                datalist[l][sz]=1

    datalist={4:
    {'512x512': 484, '384x512': 37, '512x382': 20, '512x289': 10, '512x341': 9, '512x384': 92, '512x288': 15,
     '512x340': 2, '512x431': 0, '289x512': 2, '382x512': 16, '512x383': 4, '341x512': 2, '511x512': 4, '512x343': 2,
     '510x512': 0, '512x339': 1, '381x512': 0, '512x307': 1, '339x512': 0, '512x342': 5, '512x306': 4, '512x304': 0,
     '512x385': 1, '499x512': 0, '512x483': 0, '512x326': 0, '359x512': 0, '440x512': 0, '512x510': 0, '512x508': 0,
     '512x471': 0, '383x512': 0, '287x512': 0, '512x509': 0, '512x315': 0, '512x287': 0, '512x308': 0, '306x512': 0},
    7:
    {'384x512': 45, '512x384': 111, '512x512': 473, '512x341': 15, '512x511': 1, '512x383': 8, '512x343': 1,
     '512x378': 0, '512x508': 0, '512x484': 0, '512x307': 5, '512x382': 11, '512x390': 0, '512x340': 3, '512x288': 9,
     '512x385': 1, '512x306': 7, '512x334': 0, '512x461': 0, '511x512': 4, '512x323': 0, '306x512': 0, '512x361': 0,
     '512x342': 3, '512x289': 3, '453x512': 0, '344x512': 0, '467x512': 0, '512x452': 0, '512x345': 0, '382x512': 6,
     '512x459': 0, '361x512': 0, '512x362': 0, '307x512': 1, '512x500': 0, '289x512': 0, '512x366': 0, '512x509': 1,
     '512x287': 0, '512x363': 0, '442x512': 0},
    6:
    {'512x512': 484, '384x512': 63, '383x512': 1, '306x512': 4, '512x382': 12, '506x512': 0, '512x289': 9, '510x512': 0,
     '512x307': 4, '512x384': 87, '512x341': 6, '512x306': 3, '382x512': 13, '340x512': 0, '512x254': 0, '512x383': 5,
     '512x409': 0, '512x504': 0, '512x288': 2, '379x512': 0, '512x482': 0, '512x337': 0, '512x484': 0, '511x512': 0,
     '512x332': 0, '287x512': 0, '512x511': 0, '289x512': 2, '341x512': 2, '512x348': 0, '307x512': 0, '512x372': 0,
     '512x441': 0, '512x375': 0, '512x385': 0, '512x287': 0, '512x343': 2, '512x273': 0, '512x340': 1, '512x358': 0,
     '512x344': 0, '428x512': 0, '512x498': 0, '512x471': 0, '512x509': 0, '512x387': 0, '442x512': 0, '512x410': 0,
     '512x342': 0, '304x512': 0},
    0:
    {'308x512': 1, '512x512': 492, '512x384': 123, '340x512': 1, '512x340': 3, '512x341': 10, '384x512': 32,
     '512x511': 2, '512x510': 1, '512x343': 3, '512x385': 1, '512x382': 9, '512x506': 0, '289x512': 1, '382x512': 9,
     '511x512': 4, '339x512': 0, '512x477': 0, '512x383': 3, '426x512': 0, '512x459': 0, '512x339': 0, '512x306': 2,
     '512x342': 3, '512x288': 6, '512x301': 1, '512x474': 0, '306x512': 0, '512x289': 1, '512x326': 0, '512x287': 0,
     '512x417': 0, '509x512': 0, '512x232': 0, '512x256': 0, '512x481': 0, '512x472': 0, '512x442': 0, '512x487': 0,
     '512x307': 0, '512x480': 0, '341x512': 0},
    1:
    {'512x384': 119, '512x512': 465, '384x512': 37, '304x512': 0, '512x471': 0, '512x382': 17, '512x289': 10,
     '512x341': 12, '512x306': 8, '512x511': 1, '306x512': 2, '382x512': 10, '512x500': 0, '289x512': 1, '512x288': 3,
     '512x345': 0, '512x383': 4, '512x470': 0, '512x307': 3, '512x291': 0, '389x512': 0, '512x508': 1, '341x512': 1,
     '512x287': 2, '307x512': 0, '512x340': 1, '512x385': 1, '288x512': 2, '512x342': 2, '512x365': 0, '512x393': 0,
     '512x332': 0, '512x381': 0, '512x506': 0, '512x477': 1, '511x512': 1, '512x304': 0, '404x512': 0, '340x512': 0,
     '512x479': 0, '512x308': 0, '383x512': 3, '328x512': 0},
    8:
    {'512x512': 473, '512x384': 102, '384x512': 46, '503x512': 0, '512x471': 0, '388x512': 0, '512x307': 3,
     '512x306': 12, '512x289': 8, '512x288': 8, '511x512': 4, '382x512': 13, '512x341': 12, '412x512': 0, '512x340': 3,
     '306x512': 6, '339x512': 0, '512x382': 11, '512x376': 0, '289x512': 0, '512x385': 0, '512x342': 2, '512x366': 0,
     '512x383': 3, '402x512': 0, '383x512': 0, '288x512': 0, '477x512': 0, '512x480': 0, '307x512': 0, '512x380': 0,
     '308x512': 0, '512x381': 0, '512x499': 0, '341x512': 0, '512x511': 0, '381x512': 0, '512x496': 0, '512x343': 0,
     '512x490': 0, '512x418': 0, '512x403': 0, '512x508': 0, '340x512': 0},
    2:
    {'511x512': 8, '512x306': 9, '512x512': 450, '512x384': 113, '512x342': 3, '342x512': 2, '512x341': 18,
     '512x289': 11, '512x339': 0, '512x340': 5, '512x383': 0, '512x302': 0, '306x512': 1, '384x512': 42, '339x512': 0,
     '382x512': 15, '512x382': 9, '512x288': 6, '512x349': 0, '512x307': 3, '512x492': 1, '512x395': 0, '512x324': 0,
     '507x512': 0, '509x512': 0, '512x511': 0, '512x344': 0, '512x310': 0, '512x343': 1, '512x236': 0, '512x363': 0,
     '510x512': 0, '307x512': 0, '325x512': 0, '512x322': 0, '512x439': 0, '512x476': 0, '512x385': 1, '289x512': 0,
     '288x512': 2, '512x508': 0, '512x505': 0, '341x512': 1, '380x512': 0, '446x512': 0, '512x404': 0, '512x369': 0,
     '512x471': 0, '512x386': 0},
    5:
    {'512x512': 525, '512x341': 10, '306x512': 3, '512x340': 5, '512x493': 0, '512x382': 17, '512x384': 80,
     '289x512': 2, '512x344': 0, '382x512': 23, '511x512': 3, '512x287': 2, '512x349': 0, '512x288': 5, '288x512': 0,
     '384x512': 23, '307x512': 0, '341x512': 2, '512x509': 0, '512x339': 2, '512x342': 1, '512x326': 0, '512x440': 0,
     '512x383': 4, '512x479': 0, '512x418': 0, '512x489': 0, '512x289': 3, '381x512': 0, '512x306': 4, '512x410': 0,
     '389x512': 0, '420x512': 0, '512x307': 0, '383x512': 0, '512x465': 0},
    3:
    {'512x384': 92, '512x341': 10, '512x366': 0, '384x512': 32, '512x512': 496, '382x512': 21, '306x512': 3,
     '512x306': 5, '289x512': 2, '511x512': 6, '512x231': 0, '512x342': 2, '512x340': 3, '512x277': 0, '512x307': 2,
     '512x382': 12, '512x386': 0, '512x490': 0, '512x289': 3, '512x269': 0, '512x339': 0, '512x383': 4, '339x512': 0,
     '512x288': 8, '512x511': 1, '287x512': 1, '501x512': 0, '340x512': 0, '341x512': 1, '512x388': 1, '512x298': 0,
     '510x512': 0, '512x344': 0, '512x287': 0, '512x236': 0, '383x512': 0, '512x393': 0, '512x369': 0, '512x325': 0,
     '512x343': 0, '512x326': 0, '354x512': 0, '512x509': 0, '512x286': 0, '512x456': 0},
    9:
    {'512x512': 484, '512x384': 89, '512x382': 24, '512x289': 8, '382x512': 26, '512x510': 0, '384x512': 37,
     '512x341': 5, '512x455': 0, '512x306': 10, '479x512': 0, '512x449': 0, '512x383': 5, '512x500': 0, '512x288': 7,
     '512x307': 2, '512x347': 0, '507x512': 0, '512x511': 1, '306x512': 4, '289x512': 1, '512x259': 0, '512x298': 0,
     '512x319': 0, '511x512': 1, '512x404': 0, '288x512': 1, '512x304': 1, '512x342': 3, '512x430': 0, '512x287': 1,
     '341x512': 0, '512x367': 0, '512x508': 0, '512x316': 0, '512x385': 0, '383x512': 1, '512x340': 0, '510x512': 0}}
    for k,v in datalist.items():
        print(k,v)



def ResultAnalysis():
    base_path="evaluation/Results/"
    for f in os.listdir(base_path):
        df=pd.read_csv(base_path+f)
        # df=df.pivot_table(values="Epoch",columns=[str(i) for i in range(150)])
        df=df.melt(id_vars='Epoch', var_name='Column', value_name='Value')
        df['Column']=df['Column'].astype(int)
        df=df.pivot_table(index='Column', columns='Epoch', values='Value')
        print('x'.join(f.split(".")[0].split('_')[-2:]),"%.3f"%(df["Train"].max()),"%.3f"%(df["Test"].max()))


# DataAnalysis()
ResultAnalysis()

# https://github.com/idealo/image-super-resolution