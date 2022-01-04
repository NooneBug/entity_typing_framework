from pytorch_lightning.core.lightning import LightningModule

class OneHotTypeEncoder(LightningModule):
    '''
    since the datasetmanager represent true labels as one hot, this is a fake module, only returns the already correct labels
    '''
    def __init__(self, type_number, trainable = False):
        super().__init__()

    def forward(self, batched_labels):
        return batched_labels