from torch.nn import BCELoss


class Loss():
    def __init__(self) -> None:
        pass

    def compute_loss():
        raise Exception('Implement this function')

class BCELossForET(Loss):
    def __init__(self,loss_params) -> None:
        super().__init__()
        self.loss = BCELoss(loss_params)
    
    def compute_loss(self, encoded_input, type_representation):
        return self.loss(encoded_input, type_representation)