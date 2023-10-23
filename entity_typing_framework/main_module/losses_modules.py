from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
from pytorch_lightning.core.lightning import LightningModule
# from torchmetrics import LabelRankingLoss
from entity_typing_framework.utils.flat2hierarchy import get_type2id_original, get_descendants_map, get_loss_input
import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import CrossEntropyLoss, KLDivLoss
from collections import defaultdict
from itertools import combinations
from torch.nn.functional import cosine_similarity, log_softmax

from fast_soft_sort.pytorch_ops import soft_rank

from typing import Any, Optional
from torch import Tensor
# from torchmetrics.functional.classification.ranking import (
#     _label_ranking_loss_compute,
#     _label_ranking_loss_update,
# )
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
        if 'pos_weight' in loss_params:
            loss_params['pos_weight'] = torch.tensor([float(loss_params['pos_weight'])] * len(type2id))
        self.loss = IMPLEMENTED_CLASSES_LVL1[name](**loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation.to(torch.float32))

class WeightedBCELossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params)
        name = loss_params.pop('name')

        id2types = {v: k for k, v in type2id.items()}

        self.type_weights = torch.tensor([float(len(id2types[i].split('/')) - 1) for i, t in enumerate(type2id)], device=self.device)

        self.loss = IMPLEMENTED_CLASSES_LVL1[name](reduction = 'none', **loss_params)

    def compute_loss(self, encoded_input, type_representation):
        intermediate_losses = self.loss(encoded_input, type_representation.to(torch.float32))
        final_loss = torch.mean(self.type_weights * intermediate_losses)

        return final_loss

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

class CELossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params)
        name = loss_params.pop('name')
        self.loss = IMPLEMENTED_CLASSES_LVL1[name](**loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        argmax = type_representation.argmax(axis = -1)
        return self.loss(encoded_input, argmax)
    
class KLDivLossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params)
        name = loss_params.pop('name')
        self.loss = IMPLEMENTED_CLASSES_LVL1[name](**loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        log_input = log_softmax(encoded_input, dim=-1)
        return self.loss(log_input.double(), type_representation.double())


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

                
class ALIGNIELossModule(LossModule):
    def __init__(self, type2id, loss_params, main_loss='KLDivLossModule', enable_hierarchic_losses = True, enable_verbalizer_losses = True, **kwargs) -> None:
        super().__init__(type2id, loss_params, **kwargs)
        self.main_loss = globals()[main_loss](type2id=type2id, loss_params=loss_params)
        # self.ce_loss = CELossModule(type2id=type2id, loss_params=loss_params)
        # self.kld = KLDivLossModule(type2id=type2id, loss_params=loss_params)
        self.verbalizer_loss = VerbalizerLossModule(type2id=type2id, loss_params=loss_params)
        self.hierarchic_loss = HierarchicLossModule(type2id=type2id, loss_params=loss_params)

        self.enable_hierarchic_losses = enable_hierarchic_losses
        self.enable_verbalizer_losses = enable_verbalizer_losses

    def compute_loss_for_training_step(self, encoded_input, type_representation):
        encoded_input, verbalizer_matrix, verbalizer = encoded_input
        return self.compute_loss(encoded_input=encoded_input,
                                 verbalizer_matrix=verbalizer_matrix,
                                 verbalizer=verbalizer,
                                 type_representation=type_representation,
                                 )
    
    def compute_loss_for_validation_step(self, encoded_input, type_representation):
        encoded_input, verbalizer_matrix, verbalizer = encoded_input
        return self.compute_loss(encoded_input=encoded_input,
                                 verbalizer_matrix=verbalizer_matrix,
                                 verbalizer=verbalizer,
                                 type_representation=type_representation,
                                 )

    def compute_loss(self, encoded_input, verbalizer_matrix, verbalizer, type_representation):
        # ce_loss_value = self.ce_loss.compute_loss(encoded_input, type_representation)
        main_loss_value = self.main_loss.compute_loss(encoded_input, type_representation)
        verbalizer_loss_value = 0.
        incl_loss_value, excl_loss_value = 0., 0.

        if self.enable_verbalizer_losses:
            verbalizer_loss_value = self.verbalizer_loss.compute_loss(verbalizer_matrix.weight, verbalizer)
            
        if self.enable_hierarchic_losses:
            incl_loss_value, excl_loss_value = self.hierarchic_loss.compute_loss(verbalizer_matrix.weight)
        
        return {'main_loss': main_loss_value, 
                'verbalizer_loss' : verbalizer_loss_value, 
                'incl_loss_value' : incl_loss_value, 
                'excl_loss_value' : excl_loss_value}
    
class VerbalizerLossModule(LossModule):
    def compute_loss(self, verbalizer_matrix, verbalizer):

        all_keywords = []
        keyword_class = []
        for k, value in verbalizer.items():
            all_keywords.extend(value)
            keyword_class.extend([k for j in range(len(value))])        

        keyword_class = torch.tensor(keyword_class).cuda()  
        all_keywords = torch.tensor(all_keywords).cuda()
        all_keywords = all_keywords.view(-1).repeat(len(verbalizer), 1)  
        score = torch.gather(verbalizer_matrix, 1, all_keywords).T   


        loss_fct = CrossEntropyLoss()
        disc_loss = loss_fct(score, keyword_class)
        return disc_loss

class HierarchicLossModule(LossModule):
    def __init__(self, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params, **kwargs)
        father_son_dict = defaultdict(list)

        fathers = []

        for t in type2id.keys():
            # if t.split('/other')[-1] == '':
            #     fathers.append(t.split('/other')[0])
            # else:
            divided_types = t.split('/')
            if len(divided_types) > 2:
                fathers.append(divided_types[1])

        fathers = list(set(fathers))
        for f in fathers:
            for type2 in type2id.keys():
                # if f + '/other' != type2 and f + '/' in type2:
                #     father_son_dict[type2id[f + '/other']].append(type2id[type2])
                if f + '/' in type2:
                    father_son_dict['/' + f].append(type2)
        
        self.father_son_couples = []

        for father, sons in father_son_dict.items():
            if father in type2id:
                for s in sons:
                    self.father_son_couples.append((type2id[father], type2id[s]))
        
        self.siblings_couples = []
        for sons in father_son_dict.values():
            if len(sons) > 1:
                sons = [type2id[s] for s in sons]
                self.siblings_couples.extend(list(combinations(sons, 2)))
        x=0
    
    def compute_loss(self, verbalizer_matrix):
        incl_loss_value = self.incl_loss(verbalizer_matrix)
        excl_loss_value = self.excl_loss(verbalizer_matrix)

        return incl_loss_value, excl_loss_value

    def incl_loss(self, verbalizer_matrix):

        similarities = torch.tensor(0.).cuda()
        for f, s in self.father_son_couples:
            similarities += 1 - cosine_similarity(verbalizer_matrix[f], verbalizer_matrix[s], dim = -1)

        return similarities / len(self.father_son_couples)

    def excl_loss(self, verbalizer_matrix):
        similarities = torch.tensor(0.).cuda()
        for s1, s2 in self.siblings_couples:
            similarities += cosine_similarity(verbalizer_matrix[s1], verbalizer_matrix[s2], dim = -1)

        return similarities / len(self.siblings_couples)