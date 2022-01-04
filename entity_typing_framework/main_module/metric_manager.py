import torch
import pytorch_lightning as pl
from typing import Any, NoReturn, Optional
import torchmetrics

class MetricManager():

    def __init__(self):
        self.micro_p = torchmetrics.Precision(num_classes=3, average='micro', mdmc_average='global')
        self.micro_r = torchmetrics.Recall(num_classes=3, average='micro', mdmc_average='global')
        
        self.macro_p_ex = torchmetrics.Precision(num_classes=3, average='samples', mdmc_average='global')
        self.macro_r_ex = torchmetrics.Recall(num_classes=3, average='samples', mdmc_average='global')
        
        self.macro_p_t = torchmetrics.Precision(num_classes=3, average='macro', mdmc_average='global')
        self.macro_r_t = torchmetrics.Recall(num_classes=3, average='macro', mdmc_average='global')

    def update(self, pred, target):
        pred = pred.float()
        self.micro_p.update(preds=pred, target=target)
        self.micro_r.update(preds=pred, target=target)

        self.macro_p_ex.update(preds=pred, target=target)
        self.macro_r_ex.update(preds=pred, target=target)

        self.macro_p_t.update(preds=pred, target=target)
        self.macro_r_t.update(preds=pred, target=target)
    
    def compute(self):
        micro_p = self.micro_p.compute()
        micro_r = self.micro_r.compute()
        micro_f1 = self.compute_f1(micro_p, micro_r)

        macro_p_ex = self.macro_p_ex.compute()
        macro_r_ex = self.macro_r_ex.compute()
        macro_f1_ex = self.compute_f1(macro_p_ex, macro_r_ex)

        macro_p_t = self.macro_p_t.compute()
        macro_r_t = self.macro_r_t.compute()
        macro_f1_t = self.compute_f1(macro_p_t, macro_r_t)

        return micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t

    def compute_f1(self, p, r):
        return (2 * p * r) / (p + r)