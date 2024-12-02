# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:38:22 2018

@author: lenovo
"""
import numpy as np
import torch
import pandas as pd

###=====check the acc of model on loader, if error_analysis return confuseMatrix====
def check_accuracy(model, loader, device, error_analysis=False):
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        # one batch
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix 
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix
