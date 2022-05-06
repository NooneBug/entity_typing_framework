from typing import Any, Dict
# from entity_typing_framework.main_module.inference_manager import BaseInferenceManager
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
    loss_params,
    inference_params : dict,
    checkpoint_to_load : str = None,
    ):

        super().__init__()
        self.type_number = type_number
        self.logger_module = logger

        if not checkpoint_to_load:
            self.ET_Network = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, type_number = self.type_number)
        else:
            self.ET_Network = self.load_from_checkpoint(ET_Network_params=ET_Network_params, checkpoint_to_load=checkpoint_to_load)
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device)
        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device)

        self.inference_manager = IMPLEMENTED_CLASSES_LVL0[inference_params['name']](**inference_params)
        self.loss = IMPLEMENTED_CLASSES_LVL0[loss_params['name']](**loss_params)
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

        return loss
    
    def validation_epoch_end(self, out):
        metrics = self.metric_manager.compute()
        
        self.logger_module.log_all_metrics(metrics)
        val_loss = torch.mean(torch.tensor(out))
        self.logger_module.log_loss(name = 'val_loss', value = val_loss)
        self.logger_module.log_all()
        
        self.log("val_loss", val_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def load_from_checkpoint(self, ET_Network_params, checkpoint_to_load):
        return IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number = self.type_number).load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, 
                                                                                                                            strict = False,
                                                                                                                            **ET_Network_params)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'input_projector' in k}
        del checkpoint['hyper_parameters']['logger']
        return super().on_save_checkpoint(checkpoint)

    def on_train_end(self) -> None:
        self.predict_on_test()

    def predict_on_test(self):
        test_dataloader = self.trainer.datamodule.test_dataloader()
        for batch in test_dataloader:
            _, _, true_types = batch
            network_output, type_representations = self.ET_Network(batch)
            inferred_types = self.inference_manager.infer_types(network_output)
            self.test_metric_manager.update(inferred_types, true_types)

        test_metrics = self.test_metric_manager.compute()
        self.logger_module.log_all_metrics(test_metrics)
        self.logger_module.log_all()


class IncrementalMainModule(MainModule):

    def __init__(self, ET_Network_params: dict, type_number: int, logger, loss_params, checkpoint_to_load: str = None, new_type_number = None):
        self.new_type_number = new_type_number
        super().__init__(ET_Network_params, type_number, logger, loss_params, checkpoint_to_load)
        
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='pretraining_')
        self.incremental_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='incremental_')

        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_pretraining_')
        self.test_incremental_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_incremental_')


    def on_fit_start(self):
        self.metric_manager.set_device(self.device)
        self.incremental_metric_manager.set_device(self.device)
        self.test_metric_manager.set_device(self.device)
        self.test_incremental_metric_manager.set_device(self.device)

    def load_from_checkpoint(self, ET_Network_params, checkpoint_to_load):
        ckpt_state_dict = torch.load(checkpoint_to_load)
        checkpoint_type_number = ckpt_state_dict['hyper_parameters']['type_number']
        checkpoint = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number = checkpoint_type_number).load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, 
                                                                                                                            strict = False,
                                                                                                                            **ET_Network_params)
        if not self.new_type_number:
            self.new_type_number = self.type_number - checkpoint_type_number
        
        checkpoint.setup_incremental_training(new_type_number = self.new_type_number, network_params = ET_Network_params['network_params'])
        return checkpoint

    def training_epoch_end(self, out):
        losses = [v for elem in out for k, v in elem.items()]
        self.logger_module.log_loss(name = 'losses/train_loss', value = torch.mean(torch.tensor(losses)))

    def training_step(self, batch, batch_step):
        pretraining_batch, incremental_batch = batch['pretraining'], batch['incremental']
        loss = 0
        for minibatch in [pretraining_batch, incremental_batch]: 
            network_output, type_representations = self.ET_Network(minibatch)
            l = self.loss.compute_loss(network_output, type_representations)
            loss += l
        return loss


    def validation_step(self, batch, batch_step):
        pretraining_batch, incremental_batch = batch['pretraining'], batch['incremental']
        pretraining_val_loss = 0
        incremental_val_loss = 0
        for name, minibatch in zip(['pretraining', 'incremental'], [pretraining_batch, incremental_batch]): 
            _, _, true_types = minibatch
            network_output, type_representations = self.ET_Network(minibatch)
            loss = self.loss.compute_loss(network_output, type_representations)
            inferred_types = self.inference_manager.infer_types(network_output)
            if name == 'pretraining':
                self.metric_manager.update(inferred_types, true_types)
                pretraining_val_loss += loss
            else:
                self.incremental_metric_manager.update(inferred_types, true_types)
                incremental_val_loss += loss

            self.log("losses/{}_val_loss".format(name), loss)
        val_loss = pretraining_val_loss + incremental_val_loss
        self.log("losses/val_loss".format(name), val_loss)
        return val_loss, pretraining_val_loss, incremental_val_loss
    
    def validation_epoch_end(self, out):
        metrics = self.metric_manager.compute()
        self.logger_module.log_all_metrics(metrics)
        incremental_metrics = self.incremental_metric_manager.compute()
        self.logger_module.log_all_metrics(incremental_metrics)
        self.logger_module.log_loss(name = 'losses/val_loss', value = torch.mean(torch.tensor(out)))
        self.logger_module.log_all()

    def on_train_end(self) -> None:
        self.predict_on_test()

    def predict_on_test(self):
        test_dataloader = self.trainer.datamodule.test_dataloader()
        for batch in test_dataloader:
            pretraining_batch, incremental_batch = batch['pretraining'], batch['incremental']
            for name, minibatch in zip(['pretraining', 'incremental'], [pretraining_batch, incremental_batch]): 
                minibatch = [elem.cuda() for elem in minibatch]
                _, _, true_types = minibatch
                network_output, type_representations = self.ET_Network(minibatch)
                loss = self.loss.compute_loss(network_output, type_representations)
                inferred_types = self.inference_manager.infer_types(network_output)
                if name == 'pretraining':
                    self.test_metric_manager.update(inferred_types, true_types)
                else:
                    self.test_incremental_metric_manager.update(inferred_types, true_types)

        test_metrics = self.test_metric_manager.compute()
        self.logger_module.log_all_metrics(test_metrics)
        test_incremental_metrics = self.test_incremental_metric_manager.compute()
        self.logger_module.log_all_metrics(test_incremental_metrics)
        self.logger_module.log_all()


        

class KENNMainModule(MainModule):
    pass    


class KENNMultilossMainModule(KENNMainModule):
    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, type_representations)
        # pick postkenn predictions
        inferred_types = self.inference_manager.infer_types(network_output[1])
        self.metric_manager.update(inferred_types, true_types)
        self.log("val_loss", loss)

class IncrementalKENNMainModule(IncrementalMainModule):
    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        loss = self.loss.compute_loss(network_output, type_representations)
        # pick postkenn predictions
        inferred_types = self.inference_manager.infer_types(network_output[1])
        self.metric_manager.update(inferred_types, true_types)
        self.log("val_loss", loss)
