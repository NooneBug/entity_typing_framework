from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
from pytorch_lightning.core.lightning import LightningModule
# from torchmetrics import LabelRankingLoss
from entity_typing_framework.utils.flat2hierarchy import get_type2id_original, get_descendants_map, get_loss_input
import torch
from torch.nn.modules.loss import _WeightedLoss


from fast_soft_sort.pytorch_ops import soft_rank

from typing import Any, Optional
from torch import Tensor
from torchmetrics.functional.classification.ranking import (
    _label_ranking_loss_compute,
    _label_ranking_loss_update,
)
from torchmetrics.metric import Metric


class LossModule(LightningModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__()
        self.type2id = type2id

    def compute_loss_for_training_step(self, **kwargs):
        return self.compute_loss(**kwargs)
    
    def compute_loss_for_validation_step(self, **kwargs):
        return self.compute_loss(**kwargs)

    def compute_loss(self, **kwargs):
        raise Exception('Implement this function')

class BCELossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params)
        name = loss_params.pop('name')
        self.loss = IMPLEMENTED_CLASSES_LVL1[name](**loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)

# class LabelRankingLoss(Metric):
#     """Computes the label ranking loss for multilabel data [1]. The score is corresponds to the average number of
#     label pairs that are incorrectly ordered given some predictions weighted by the size of the label set and the
#     number of labels not in the label set. The best score is 0.

#     Args:
#         kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

#     Example:
#         >>> from torchmetrics import LabelRankingLoss
#         >>> _ = torch.manual_seed(42)
#         >>> preds = torch.rand(10, 5)
#         >>> target = torch.randint(2, (10, 5))
#         >>> metric = LabelRankingLoss()
#         >>> metric(preds, target)
#         tensor(0.4167)

#     References:
#         [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
#         knowledge discovery handbook (pp. 667-685). Springer US.
#     """

#     loss: Tensor
#     numel: Tensor
#     sample_weight: Tensor
#     higher_is_better: bool = False
#     is_differentiable: bool = True

#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)
#         self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("sample_weight", torch.tensor(0.0), dist_reduce_fx="sum")

#     def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:  # type: ignore
#         """
#         Args:
#             preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
#                 of labels. Should either be probabilities of the positive class or corresponding logits
#             target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
#                 of labels. Should only contain binary labels.
#             sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
#                 should be weighted in the final score.
#         """
#         loss, numel, sample_weight = _label_ranking_loss_update(preds, target, sample_weight)
#         self.loss += loss
#         self.numel += numel
#         if sample_weight is not None:
#             self.sample_weight += sample_weight

#     def compute(self) -> Tensor:
#         """Computes the label ranking loss."""
#         return _label_ranking_loss_compute(self.loss, self.numel, self.sample_weight)

class RankingLoss(_WeightedLoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, target):
        # _check_ranking_input(preds, target, self.sample_weight)
        n_preds, n_labels = preds.shape
        relevant = target == 1
        n_relevant = relevant.sum(dim=1)

        # Ignore instances where number of true labels is 0 or n_labels
        mask = (n_relevant > 0) & (n_relevant < n_labels)
        preds = preds[mask]
        relevant = relevant[mask]
        n_relevant = n_relevant[mask]

        # Nothing is relevant
        if len(preds) == 0:
            return torch.tensor(0.0, device=preds.device), 1, self.sample_weight

        inverse = soft_rank(preds.cpu(), regularization_strength=.001) - 1
        inverse = inverse.cuda()

        # inverse = preds.argsort(dim=1).argsort(dim=1)
        per_label_loss = ((n_labels - inverse) * relevant).to(torch.float32)
        correction = 0.5 * n_relevant * (n_relevant + 1)
        denom = n_relevant * (n_labels - n_relevant)
        loss = (per_label_loss.sum(dim=1) - correction) / denom
        # if isinstance(self.sample_weight, Tensor):
            # loss *= self.sample_weight[mask]
            # sample_weight = self.sample_weight.sum()
        # return loss.sum(), n_preds, sample_weight
        return loss.sum()

class RankingLossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params)
        self.loss = RankingLoss()
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)

class FlatRankingLossModule(RankingLossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params, **kwargs)
        self.type2id_flat = type2id
        self.type2id_original = get_type2id_original(type2id)
        self.descendants_map_original = get_descendants_map(self.type2id_original)
    
    def compute_loss_for_validation_step(self, encoded_input, type_representation):
        encoded_input_original, type_representation_original = get_loss_input(encoded_input, type_representation, self)
        return self.compute_loss(encoded_input_original, type_representation_original)


class FlatBCELossModule(BCELossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params, **kwargs)
        self.type2id_flat = type2id
        self.type2id_original = get_type2id_original(type2id)
        self.descendants_map_original = get_descendants_map(self.type2id_original)
    
    def compute_loss_for_validation_step(self, encoded_input, type_representation):
        encoded_input_original, type_representation_original = get_loss_input(encoded_input, type_representation, self)
        return self.compute_loss(encoded_input_original, type_representation_original)

                
            

