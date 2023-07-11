import torch
from torch import nn, optim
import torch.nn.functional as F

def ssim_loss(img1, img2, window_size=11):
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
    return 1 - SSIM.mean()


def getLoss(name):
    if name=="mse":
        return torch.nn.MSELoss()
    elif name == "ssim":
        return ssim_loss
    else:
        print(name,"Loss not supported")
