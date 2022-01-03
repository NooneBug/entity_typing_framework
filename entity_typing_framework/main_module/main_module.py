from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork
from entity_typing_framework.EntityTypingNetwork_classes.losses import Loss
from entity_typing_framework.dataset_classes.dataset_managers import DatasetManager
from pytorch_lightning.core.lightning import LightningModule
import torch

class MainModule(LightningModule):
    def __init__(self, dataset_manager : DatasetManager
    , ET_Network_params, loss : Loss
    ):

        super().__init__()
        self.dataset_manager = dataset_manager
        self.ET_Network = BaseEntityTypingNetwork(**ET_Network_params['init_args'], type_number = dataset_manager.get_type_number())
        self.loss = loss
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_step):
        network_output, label_representation = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, label_representation)
        # self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss
        
    def train_dataloader(self):
        return self.dataset_manager.dataloaders['train']
    
    def val_dataloader(self):
        return self.dataset_manager.dataloaders['dev']

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

