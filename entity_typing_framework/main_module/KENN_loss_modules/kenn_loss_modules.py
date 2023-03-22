from entity_typing_framework.main_module.losses_modules import BCELossModule
import torch

class BCEMultiLossModule(BCELossModule):
    def __init__(self, alpha, type2id, loss_params, **kwargs) -> None:
        super().__init__(type2id, loss_params, **kwargs)
        self.alpha = alpha
    
    def compute_loss(self, encoded_input, type_representation):
        prekenn, postkenn = encoded_input
        type_representation = type_representation.to(torch.float32)
        return self.alpha * self.loss(prekenn, type_representation) + (1 - self.alpha) * self.loss(postkenn, type_representation)
