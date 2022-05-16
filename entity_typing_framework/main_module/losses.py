from typing import Optional
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import BCELoss
import torch.nn as nn
import torch

class Loss(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def compute_loss():
        raise Exception('Implement this function')

class BCELossForET(Loss):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.loss = BCELoss(**kwargs)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)

class BoxEmbeddingBCELoss(Loss):
    def __init__(self, name):
        super().__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.sigmoid_fn = nn.Sigmoid()

    def compute_loss(self,
                    logits: torch.Tensor,
                    targets: torch.Tensor,
                    weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            loss = self.loss_func(logits, targets, weight=weight)
        else:
            loss = self.loss_func(logits, targets)
        return loss