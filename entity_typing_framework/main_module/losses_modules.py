from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
from pytorch_lightning.core.lightning import LightningModule

class LossModule(LightningModule):
    def __init__(self, name, loss_params) -> None:
        super().__init__()

    def compute_loss():
        raise Exception('Implement this function')

class BCELossModule(LossModule):
    def __init__(self, name, loss_params) -> None:
        super().__init__(name, loss_params)
        self.loss = IMPLEMENTED_CLASSES_LVL1[loss_params['name']](**loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)