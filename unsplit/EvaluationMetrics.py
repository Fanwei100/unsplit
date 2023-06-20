from collections import OrderedDict

import torch
from torch import nn, optim
from ignite.engine import *
from ignite.metrics import *

# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 4))
]))
vggmodel = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).features

def calulateSSIM(preds, target):
    default_evaluator = Engine(eval_step)
    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')
    state = default_evaluator.run([[preds, target]])
    return state.metrics['ssim']

def calculatFID(y_pred,y_true):
    default_evaluator = Engine(eval_step)
    metric = FID(num_features=1000, feature_extractor=vggmodel)
    metric.attach(default_evaluator, "fid")
    print(y_true.shape,y_pred.shape)
    state = default_evaluator.run([[y_pred, y_true]])
    print(state.metrics["fid"])


preds = torch.rand([1, 1, 28, 28])
target = preds * 0.75
print(calulateSSIM(preds, target))
print(calculatFID(target,preds))
