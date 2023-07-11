import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from .LossFunctions import getLoss
from .util import *




def model_inversion_stealing(clone_model, split_layer, target, input_size,lossname="mse",
                            lambda_tv=0.1, lambda_l2=1, main_iters=1000, input_iters=100, model_iters=100,device=torch.device("cpu")):
    x_pred = torch.empty(input_size).to(device).fill_(0.5).requires_grad_(True)
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    model_opt = torch.optim.Adam(clone_model.parameters(), lr=0.001, amsgrad=True)
    lossf = getLoss(lossname)
    clone_model.to(device)
    target=target.to(device)
    inputloss,clonemodelloss=[],[]
    for main_iter in tqdm(range(main_iters)):
        iloss,mloss=0,0
        for input_iter in range(input_iters):
            input_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = lossf(pred, target) + lambda_tv*TV(x_pred).to(device) + lambda_l2*l2loss(x_pred)
            loss.backward(retain_graph=True)
            input_opt.step()
            iloss+=loss
        for model_iter in range(model_iters):
            model_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = lossf(pred, target)
            loss.backward(retain_graph=True)
            model_opt.step()
            mloss += loss
        iloss,mloss=iloss.detach().cpu().numpy(),mloss.detach().cpu().numpy()
        inputloss.append(iloss/input_iters)
        clonemodelloss.append(mloss/model_iters)

    return x_pred.detach(),inputloss,clonemodelloss


def label_inference(pred, clone_model, target_grad, label_vals, grad_index):
    pred_losses = [torch.nn.CrossEntropyLoss()(pred, cd_label) for cd_label in label_vals]
    pred_grads = [torch.autograd.grad(loss, clone_model.parameters(), allow_unused=True, retain_graph=True)[grad_index] for loss in pred_losses]
    grad_losses = [torch.nn.MSELoss()(pred_grad, target_grad) for pred_grad in pred_grads]
    return torch.LongTensor([grad_losses.index(min(grad_losses))])