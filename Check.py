# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
import json
import os

import pandas as pd
import tqdm,cv2
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from ignite.engine import *
from ignite.metrics import *
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
from image_similarity_measures.quality_metrics import rmse, psnr,sre,ssim,fsim,issm,uiq,sam


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
    # print(img1.shape,mu1.shape)

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


def calulateSSIM(preds, target):
    default_evaluator = Engine(eval_step)
    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')
    state = default_evaluator.run([[preds, target]])
    return state.metrics['ssim']

import functools
from scipy.ndimage import uniform_filter
from skimage._shared import utils
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.arraycrop import crop
from skimage.util.dtype import dtype_range
def structural_similarity(im1, im2,
                          *,
                          win_size=None, gradient=False, data_range=None,
                          channel_axis=None,
                          gaussian_weights=False, full=False, **kwargs):
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    if channel_axis is not None:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    channel_axis=None,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = np.empty(nch, dtype=float_type)
        if gradient:
            G = np.empty(im1.shape, dtype=float_type)
        if full:
            S = np.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = structural_similarity(im1[_at(ch)],
                                              im2[_at(ch)], **args)
            if gradient and full:
                mssim[ch], G[_at(ch)], S[_at(ch)] = ch_result
            elif gradient:
                mssim[ch], G[_at(ch)] = ch_result
            elif full:
                mssim[ch], S[_at(ch)] = ch_result
            else:
                mssim[ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if (np.issubdtype(im1.dtype, np.floating) or
            np.issubdtype(im2.dtype, np.floating)):
            raise ValueError(
                'Since image dtype is floating point, you must specify '
                'the data_range parameter. Please read the documentation '
                'carefully (including the note). It is recommended that '
                'you always specify the data_range anyway.')
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtypes. Setting data_range based on im1.dtype.",
                 stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and (im1.dtype != np.uint8):
            warn("Setting data_range based on im1.dtype. " +
                 ("data_range = %.0f. " % data_range) +
                 "Please specify data_range explicitly to avoid mistakes.", stacklevel=2)

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = crop(S, pad).mean(dtype=np.float64)

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * im1
        grad += filter_func(-S / B2, **filter_args) * im2
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,**filter_args)
        grad *= (2 / im1.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


import phasepack.phasecong as pc
from image_similarity_measures.quality_metrics import _gradient_magnitude,_similarity_measure
def fsim1(org_img: np.ndarray, pred_img: np.ndarray, T1: float = 0.85, T2: float = 160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.

    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.

    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.

    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.

    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    # _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)

    alpha = (beta) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def TestSomething():
    # for i in range(50):
    #     x1,x2=np.random.random(50),np.random.random(50)
    #     print(mean_squared_error(x1,x2),mean_absolute_error(x1,x2))
    # x1,x2=np.random.randint(0,255,(64,64,3)),np.random.randint(0,255,(64,64,3))
    x1,x2=np.random.random((64,64,30)),np.random.random((64,64,30))
    # for i in range(1,30):
    #     print(np.all(x1[:,:,0]==x1[:,:,i]))
    # print(x1[:,:,0].shape)
    # print(ssim(x1,x2))
    # print(fsim(x1,x2))
    print(fsim1(x1,x2))
    # print(structural_similarity(x1, x2,win_size=11, data_range=4095))
    # print(structural_similarity(x1, x2,win_size=11, data_range=4095,channel_axis=2))
    # x1,x2=torch.rand((64,64,30)),torch.rand((64,64,30))
    # print(ssim_loss(torch.tensor(np.rollaxis(x1,2,0)),torch.tensor(np.rollaxis(x2,2,0))))




# DataAnalysis()
# ResultAnalysis()
TestSomething()
# https://github.com/idealo/image-super-resolution

