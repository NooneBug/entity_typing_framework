from entity_typing_framework.main_module.inference_manager import InferenceManager
from entity_typing_framework.main_module.losses import BCELossForET
from entity_typing_framework.main_module.metric_manager import MetricManager
from entity_typing_framework.utils.implemented_classes_lvl0 import IMPLEMENTED_CLASSES_LVL0
from pytorch_lightning.core.lightning import LightningModule
import torch

class MainModule(LightningModule):
    def __init__(self, 
    ET_Network_params : dict,
    type_number : int,
    logger,
    checkpoint_to_load : str = None,
    ):

        super().__init__()
        self.type_number = type_number
        self.logger_module = logger

        if not checkpoint_to_load:
            self.ET_Network = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, type_number = self.type_number)
        else:
            self.ET_Network = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']].load_from_checkpoint(checkpoint_to_load, **ET_Network_params)
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

    def training_epoch_end(self, out):
        losses = [v for elem in out for k, v in elem.items()]
        self.logger_module.log_loss(name = 'train_loss', value = torch.mean(torch.tensor(losses)))

    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, type_representations)
        inferred_types = self.inference_manager.infer_types(network_output)
        self.metric_manager.update(inferred_types, true_types)
        self.log("val_loss", loss)

        return loss
    
    def validation_epoch_end(self, out):
        metrics = self.metric_manager.compute()
        self.logger_module.log_all_metrics(metrics)
        self.logger_module.log_loss(name = 'val_loss', value = torch.mean(torch.tensor(out)))
        self.logger_module.log_all()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer