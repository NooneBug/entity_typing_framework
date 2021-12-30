from entity_typing_framework.dataset_classes.dataset_managers import DatasetManager
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.nn.modules import Linear
from torch.utils.data.dataloader import DataLoader, Dataset

class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class MainModule(LightningModule):
    def __init__(self, dataset_manager : DatasetManager
    # , ET_Network : EntityTypingNetwork
    ):

        super().__init__()
        self.dataset_manager = dataset_manager
        # self.ET_Network = ET_Network

        self.linear = Linear(32, 32)

        self.save_hyperparameters()
    
    def training_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch
        labels = labels.cuda()
        model_output = self(mention_x, left_x, right_x)
        loss = self.compute_loss(model_output, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss
        
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 100))

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

# class Layer(LightningModule):
#     def __init__(self, units = 12) -> None:
#         super().__init__()
#         self.units = units

# class MyModel(LightningModule):
#     def __init__(self, encoder_layer: Layer, decoder_layer: Layer):
#         """Example encoder-decoder model

#         Args:
#             encoder_layers: Number of layers for the encoder
#             decoder_layers: Number of layers for each decoder block
#         """
#         super().__init__()
#         self.encoder = encoder_layer
#         self.decoder = decoder_layer


#         self.save_hyperparameters()

