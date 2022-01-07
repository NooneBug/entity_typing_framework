from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork
from entity_typing_framework.main_module.inference_manager import InferenceManager
from entity_typing_framework.main_module.losses import BCELossForET
from entity_typing_framework.main_module.metric_manager import MetricManager
from pytorch_lightning.core.lightning import LightningModule
import torch

class MainModule(LightningModule):
    def __init__(self, 
    ET_Network_params : dict,
    type_number : int,
    logger
    ):

        super().__init__()
        self.type_number = type_number
        self.logger_module = logger

        self.ET_Network = BaseEntityTypingNetwork(**ET_Network_params, type_number = self.type_number)
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device)
        self.inference_manager = InferenceManager()
        self.loss = BCELossForET()
        self.save_hyperparameters()
    
    def on_fit_start(self):
        self.metric_manager.set_device(self.device)

    def training_step(self, batch, batch_step):
        network_output, type_representations = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, type_representations)
        return loss

    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, type_representations)
        inferred_types = self.inference_manager.infer_types(network_output)
        self.metric_manager.update(inferred_types, true_types)

        return loss
    
    def validation_epoch_end(self, out):
        metrics = self.metric_manager.compute()
        self.logger_module.log_all_metrics(metrics)
    
    # def train_dataloader(self):
    #     return self.dataset_manager.dataloaders['train']
    
    # def val_dataloader(self):
    #     return self.dataset_manager.dataloaders['dev']
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer