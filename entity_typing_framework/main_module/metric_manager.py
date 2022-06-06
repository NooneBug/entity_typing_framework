from typing import Optional
from torchmetrics import Precision, Recall
import torch
from torch import Tensor, tensor
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod


class MetricManager():

    def __init__(self, num_classes, device, prefix = ''):
        self.prefix = prefix

        self.micro_p = PrecisionCustom(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        self.micro_r = RecallCustom(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        
        self.macro_p_ex = PrecisionCustom(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        self.macro_r_ex = RecallCustom(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        
        self.macro_p_t = PrecisionCustom(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)
        self.macro_r_t = RecallCustom(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)

    def set_device(self, device):
        self.micro_p = self.micro_p.to(device=device)
        self.micro_r = self.micro_r.to(device=device)

        self.macro_p_ex = self.macro_p_ex.to(device=device)
        self.macro_r_ex = self.macro_r_ex.to(device=device)

        self.macro_p_t = self.macro_p_t.to(device=device)
        self.macro_r_t = self.macro_r_t.to(device=device)

    def update(self, pred, target):
        pred = pred.float()
        target = target.int()
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

        self.reset_metrics()

        return self.compose_return(micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t)

    def reset_metrics(self):
        self.micro_p.reset()
        self.micro_r.reset()
        
        self.macro_p_ex.reset()
        self.macro_r_ex.reset()
        
        self.macro_p_t.reset()
        self.macro_r_t.reset()

    def compose_return(self, micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t):
        return {'{}/micro/precision'.format(self.prefix) : micro_p,
                '{}/micro/recall'.format(self.prefix) : micro_r,
                '{}/micro/f1'.format(self.prefix) : micro_f1,
                '{}/macro_example/precision'.format(self.prefix) :  macro_p_ex,
                '{}/macro_example/recall'.format(self.prefix) : macro_r_ex,
                '{}/macro_example/f1'.format(self.prefix) : macro_f1_ex,
                '{}/macro_types/precision'.format(self.prefix) : macro_p_t,
                '{}/macro_types/recall'.format(self.prefix) : macro_r_t,
                '{}/macro_types/f1'.format(self.prefix) : macro_f1_t}

    def compute_f1(self, p, r):
        return (2 * p * r) / (p + r)

class PrecisionCustom(Precision):
    def compute(self) -> Tensor:
        tp, fp, _, fn = self._get_final_stats()
        return self._precision_compute(tp, fp, fn, self.average, self.mdmc_reduce)
    
    def _precision_compute(
        self,
        tp: Tensor,
        fp: Tensor,
        fn: Tensor,
        average: str,
        mdmc_average: Optional[str]) -> Tensor:

        numerator = tp
        denominator = tp + fp

        return _reduce_stat_scores(
            numerator=numerator,
            denominator=denominator,
            weights=None if average != "weighted" else tp + fn,
            average=average,
            mdmc_average=mdmc_average,
        )


class RecallCustom(Recall):

    def compute(self) -> Tensor:
        tp, fp, _, fn = self._get_final_stats()
        return self._recall_compute(tp, fp, fn, self.average, self.mdmc_reduce)
    
    def _recall_compute(
        self,
        tp: Tensor,
        fp: Tensor,
        fn: Tensor,
        average: str,
        mdmc_average: Optional[str]) -> Tensor:

        numerator = tp
        denominator = tp + fn

        return _reduce_stat_scores(
            numerator=numerator,
            denominator=denominator,
            weights=None if average != AverageMethod.WEIGHTED else tp + fn,
            average=average,
            mdmc_average=mdmc_average,
        )


def _reduce_stat_scores(
        numerator: Tensor,
        denominator: Tensor,
        weights: Optional[Tensor],
        average: Optional[str],
        mdmc_average: Optional[str],
        zero_division: int = 0) -> Tensor:

    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    if weights is None:
        weights = torch.ones_like(denominator)
    else:
        weights = weights.float()

    # numerator = torch.where(zero_div_mask, tensor(float(zero_division), device=numerator.device), numerator)
    # denominator = torch.where(zero_div_mask | ignore_mask, tensor(1.0, device=denominator.device), denominator)
    # weights = torch.where(ignore_mask, tensor(0.0, device=weights.device), weights)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)

    scores = weights * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = torch.where(torch.isnan(scores), tensor(float(zero_division), device=scores.device), scores)

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()

    if average in (AverageMethod.NONE, None):
        scores = torch.where(ignore_mask, tensor(float("nan"), device=scores.device), scores)
    else:
        scores = scores.sum()

    return scores