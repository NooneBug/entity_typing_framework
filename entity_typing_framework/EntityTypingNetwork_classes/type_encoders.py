from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Embedding
import torch
from torch.nn.functional import one_hot


class OneHotTypeEncoder(LightningModule):
    '''
    since the datasetmanager represent true labels as one hot, this is a fake module, only returns the already correct labels
    '''
    def __init__(self, type_number, trainable = False):
        super().__init__()
        # self.type_embedding = Embedding(type_number, type_number)
        # self.trainable = trainable
        # self.type_number = type_number
        # self.initialize_embeddings()

    # def initialize_embeddings(self):
    #     one_hots = one_hot(torch.arange(0, self.type_number))
    #     self.type_embedding.weight.data = one_hots
    #     self.type_embedding.weight.requires_grad = self.trainable

    def forward(self, batched_labels):
        return batched_labels