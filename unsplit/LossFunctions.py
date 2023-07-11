import torch
from torch import nn, optim
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
    # ux is mean of img1
    # uy is mean of imag2
    # axy is mean of img*imag2

    mu1 = F.avg_pool2d(img1, 3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, stride=1, padding=1)

    sigma1 = F.avg_pool2d(img1 ** 2, 3, stride=1, padding=1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 3, stride=1, padding=1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, stride=1, padding=1) - mu1 * mu2

    numerator = (2 * sigma12 + alpha) * (2 * mu1 * mu2 + beta)
    denominator = (sigma1 + sigma2 + alpha) * (mu1 ** 2 + mu2 ** 2 + beta)

    fsim_map = numerator / denominator

    return (1 - fsim_map.mean()) ** gamma


def getLoss(name):
    if name=="mse":
        return torch.nn.MSELoss()
    elif name == "ssim":
        return ssim_loss
    elif name=="fsim":
        return fsim_loss
    else:
        print(name,"Loss not supported")
