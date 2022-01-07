from pytorch_lightning.core.lightning import LightningModule
from torch.nn import BCELoss


class Loss(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def compute_loss():
        raise Exception('Implement this function')

class BCELossForET(Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.loss = BCELoss(**kwargs)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)