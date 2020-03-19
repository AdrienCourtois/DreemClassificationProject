'''
Focal Loss calculation script.
'''

import torch

def focal_loss(y_pred, y_true, alpha=0.75, gamma=2):
    # y_pred: tensor [B, 1] probability prediction
    # y_true: tensor [B, 1] binary ground truth
    # --
    # Output: tensor [B] loss for each prediction of the batch


    m1 = y_true == 1
    m0 = y_true == 0

    p_t = torch.zeros(y_pred.size())
    alpha_t = torch.zeros(y_pred.size())

    if y_pred.is_cuda:
        alpha_t = alpha_t.cuda()
        p_t = p_t.cuda()

    p_t[m1] = y_pred[m1]
    p_t[m0] = 1-y_pred[m0]

    alpha_t[m1] = alpha
    alpha_t[m0] = 1-alpha

    L = - alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-10)

    return L