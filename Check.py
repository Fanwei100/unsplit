# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
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
epochs,trainpercent=10,30
print(epochs,trainpercent)
print(epochs/trainpercent*100)

# x/trainpercent*100=10*trainpercent/100
epochs,trainpercent=34,30
print(epochs,trainpercent)
print(epochs*trainpercent/100)
