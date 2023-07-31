from typing import Optional
from torchmetrics import Precision, Recall
import torch
from torch import Tensor, tensor
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
import pandas as pd


class MetricManager():

    def __init__(self, num_classes, device, type2id, prefix = ''):
        self.prefix = prefix
        self.type2id = type2id

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

class MetricManagerForIncrementalTypes():

    def __init__(self, num_classes, device, prefix = ''):
        self.prefix = prefix
        self.macro_p_t = PrecisionCustom(num_classes=num_classes, average=AverageMethod.NONE, mdmc_average='global').to(device=device)
        self.macro_r_t = RecallCustom(num_classes=num_classes, average=AverageMethod.NONE, mdmc_average='global').to(device=device)

    def set_device(self, device):
        self.macro_p_t = self.macro_p_t.to(device=device)
        self.macro_r_t = self.macro_r_t.to(device=device)

    def update(self, pred, target):
        pred = pred.float()
        target = target.int()
        self.macro_p_t.update(preds=pred, target=target)
        self.macro_r_t.update(preds=pred, target=target)
    
    def compute(self, type2id):
        macro_p_t = self.macro_p_t.compute()
        macro_r_t = self.macro_r_t.compute()
        macro_f1_t = self.compute_f1(macro_p_t, macro_r_t)

        self.reset_metrics()

        return self.compose_return(macro_p_t, macro_r_t, macro_f1_t, type2id)

    def reset_metrics(self):
        self.macro_p_t.reset()
        self.macro_r_t.reset()

    def compose_return(self, macro_p_t, macro_r_t, macro_f1_t, type2id):
        metrics = {}
        for type, id in type2id.items():
            key = f"{self.prefix}_{type[1:].replace('/','-')}"
            metrics[f'{key}/macro_types/precision'] = macro_p_t[id]
            metrics[f'{key}/macro_types/recall'] = macro_r_t[id]
            metrics[f'{key}/macro_types/f1'] = macro_f1_t[id]

        return metrics
        

    def compute_f1(self, p, r):
        return (2 * p * r) / (p + r)

class LeavesMetricManager(MetricManager):

    def __init__(self, num_classes, device, type2id, prefix=''):
        self.leaves_ids = self.get_leaves(type2id=type2id)
        
        num_classes = len(self.leaves_ids) 
        
        super().__init__(num_classes=num_classes, device=device, type2id=type2id, prefix=prefix)
    
    def get_leaves(self, type2id):
        leaves_ids = []
        for t1 in type2id:
            is_leaf = True
            for t2 in type2id:
                if t1 != t2 and t1 in t2:
                    is_leaf = False
                    break
            if is_leaf:
                leaves_ids.append(type2id[t1])
        return leaves_ids

    def update(self, pred, target):
        target = target[:, self.leaves_ids]
        return super().update(pred, target)
    
class ALIGNIEMetricManager(MetricManager):
    def __init__(self, num_classes, device, type2id, prefix = ''):
        self.prefix = prefix
        self.type2id = type2id

        self.micro_p = PrecisionCustom(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        self.micro_r = RecallCustom(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        
        self.macro_p_ex = PrecisionCustom(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        self.macro_r_ex = RecallCustom(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        
        self.macro_p_t = PrecisionCustom(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)
        self.macro_r_t = RecallCustom(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)
        
        self.strat_p = PrecisionCustom(num_classes=num_classes, average=None,  mdmc_average='global').to(device=device)
        self.strat_r = RecallCustom(num_classes=num_classes, average=None, mdmc_average='global').to(device=device)

    def set_device(self, device):
        self.micro_p = self.micro_p.to(device=device)
        self.micro_r = self.micro_r.to(device=device)

        self.macro_p_ex = self.macro_p_ex.to(device=device)
        self.macro_r_ex = self.macro_r_ex.to(device=device)

        self.macro_p_t = self.macro_p_t.to(device=device)
        self.macro_r_t = self.macro_r_t.to(device=device)
        
        self.strat_p = self.strat_p.to(device=device)
        self.strat_r = self.strat_r.to(device=device)

    def update(self, pred, target):
        pred = pred.float()
        target = target.int()
        self.micro_p.update(preds=pred, target=target)
        self.micro_r.update(preds=pred, target=target)

        self.macro_p_ex.update(preds=pred, target=target)
        self.macro_r_ex.update(preds=pred, target=target)

        self.macro_p_t.update(preds=pred, target=target)
        self.macro_r_t.update(preds=pred, target=target)
        
        self.strat_p.update(preds=pred, target=target)
        self.strat_r.update(preds=pred, target=target)
    
    def compute(self):
        micro_p = self.micro_p.compute()
        micro_r = self.micro_r.compute()
        self.micro_p_val = torch.tensor([micro_p]).cpu()
        self.micro_r_val = torch.tensor([micro_r]).cpu()
        micro_f1 = self.compute_f1(micro_p, micro_r)

        macro_p_ex = self.macro_p_ex.compute()
        macro_r_ex = self.macro_r_ex.compute()
        self.macro_p_ex_val = torch.tensor([macro_p_ex]).cpu()
        self.macro_r_ex_val = torch.tensor([macro_r_ex]).cpu()
        macro_f1_ex = self.compute_f1(macro_p_ex, macro_r_ex)

        macro_p_t = self.macro_p_t.compute()
        macro_r_t = self.macro_r_t.compute()
        self.macro_p_t_val = torch.tensor([macro_p_t]).cpu()
        self.macro_r_t_val = torch.tensor([macro_r_t]).cpu()
        macro_f1_t = self.compute_f1(macro_p_t, macro_r_t)
        
        self.strat_p_val = self.strat_p.compute().cpu()
        self.strat_r_val = self.strat_r.compute().cpu()
        
        self.reset_metrics()

        return self.compose_return(micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t)
    
    def df_metrics(self):
        final_p = torch.cat((self.strat_p_val, self.micro_p_val, self.macro_p_ex_val, self.macro_p_t_val)) 
        final_r = torch.cat((self.strat_r_val, self.micro_r_val, self.macro_r_ex_val, self.macro_r_t_val))
        final_f1 = torch.tensor([self.compute_f1(final_p[i], final_r[i]) for i in range(final_p.shape[0])])
        labels = list(self.type2id.keys()) + ['micro_avg', 'macro_ex_avg', 'macro_t_avg']
        
        return pd.DataFrame({'label':labels, 
                             'precision':final_p.cpu().numpy(), 
                             'recall':final_r.cpu().numpy(),
                             'f1-score': final_f1.cpu().numpy()})

    def reset_metrics(self):
        self.micro_p.reset()
        self.micro_r.reset()
        
        self.macro_p_ex.reset()
        self.macro_r_ex.reset()
        
        self.macro_p_t.reset()
        self.macro_r_t.reset()
        
        self.strat_p.reset()
        self.strat_r.reset()

class PrecisionCustom(Precision):
    def compute(self) -> Tensor:
        tp, fp, _, fn = self._get_final_stats()
        return self._precision_compute(tp, fp, fn, self.average, self.mdmc_reduce)
    
    def _get_final_stats(self):
        """Performs concatenation on the stat scores if neccesary, before passing them to a compute function."""
        if isinstance(self.tp, list):
            tp = torch.cat(self.tp) if len(self.tp) > 0 else torch.tensor(0)
        else:
            tp = self.tp

        if isinstance(self.fp, list):
            fp = torch.cat(self.fp) if len(self.fp) > 0 else torch.tensor(0)
        else:
            fp = self.fp

        if isinstance(self.tn, list):
            tn = torch.cat(self.tn) if len(self.tn) > 0 else torch.tensor(0)
        else:
            tn = self.tn
        
        if isinstance(self.fn, list):
            fn = torch.cat(self.fn) if len(self.fn) > 0 else torch.tensor(0)
        else:
            fn = self.fn

        return tp, fp, tn, fn

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
    
    def _get_final_stats(self):
        """Performs concatenation on the stat scores if neccesary, before passing them to a compute function."""
        if isinstance(self.tp, list):
            tp = torch.cat(self.tp) if len(self.tp) > 0 else torch.tensor(0)
        else:
            tp = self.tp

        if isinstance(self.fp, list):
            fp = torch.cat(self.fp) if len(self.fp) > 0 else torch.tensor(0)
        else:
            fp = self.fp

        if isinstance(self.tn, list):
            tn = torch.cat(self.tn) if len(self.tn) > 0 else torch.tensor(0)
        else:
            tn = self.tn
        
        if isinstance(self.fn, list):
            fn = torch.cat(self.fn) if len(self.fn) > 0 else torch.tensor(0)
        else:
            fn = self.fn

        return tp, fp, tn, fn

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