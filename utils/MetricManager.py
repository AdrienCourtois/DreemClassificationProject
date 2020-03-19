'''
MetricManager class
Calculates the different needed metrics in an online fashion.

Usage:
metric = MetricManager()
metric.update(y_pred, y_true)
print(metric.accuracy)
metric.reset()

Note:
This is not a PyTorch Dataset.
'''

import torch

class MetricManager:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.TP = self.FP = 0
        self.TN = self.FN = 0
        self.N = 0
    
    def update(self, y_pred, y_true):
        # Updates the internal statistics given y_pred and y_true.
        # Args:
        # y_pred (numpy or torch.Tensor of shape (B,)) contains the binary prediction
        # y_true (numpy or torch.Tensor of shape (B,)) contains the ground truth

        a = (y_pred[y_true == 1] == 1).sum()
        b = (y_pred[y_true == 0] == 0).sum()
        c = (y_pred[y_true == 0] == 1).sum()
        d = (y_pred[y_true == 1] == 0).sum()

        if torch.is_tensor(y_pred):
            a, b = a.item(), b.item()
            c, d = c.item(), d.item()
        
        self.TP += a
        self.TN += b
        self.FP += c
        self.FN += d

        self.N += len(y_pred)
    
    @property
    def f1(self):
        # The current f1-score.

        P, R = self.precision, self.recall

        if P + R > 0:
            return 2 * P * R / (P + R)
        
        return 0
    
    @property
    def precision(self):
        # The current precision.

        if self.TP + self.FP > 0:
            return self.TP / (self.TP + self.FP)
        
        return 0
    
    @property
    def recall(self):
        # The current recall.

        if self.TP + self.FN > 0:
            return self.TP / (self.TP + self.FN)
        
        return 0
    
    @property
    def accuracy(self):
        # The current accuracy.

        if self.N > 0:
            return (self.TP + self.TN) / self.N
        
        return -1