from tabnanny import check
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
    type2id : dict,
    logger,
    loss_params,
    inference_params : dict,
    checkpoint_to_load : str = None,
    avoid_sanity_logging : bool = False,
    smart_save : bool = True
    ):

        super().__init__()
        self.type_number = type_number
        self.logger_module = logger
        self.ET_Network_params = ET_Network_params
        self.type2id = type2id
        self.avoid_sanity_logging = avoid_sanity_logging
        self.smart_save = smart_save

        if not checkpoint_to_load:
            self.ET_Network = IMPLEMENTED_CLASSES_LVL0[self.ET_Network_params['name']](**self.ET_Network_params, type_number = self.type_number, type2id = self.type2id)
        else:
            self.ET_Network = self.load_ET_Network(ET_Network_params=self.ET_Network_params, checkpoint_to_load=checkpoint_to_load)
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='dev')
        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test')

        self.inference_manager = IMPLEMENTED_CLASSES_LVL0[inference_params['name']](**inference_params)
        self.loss = IMPLEMENTED_CLASSES_LVL0[loss_params['name']](**loss_params)
        self.save_hyperparameters()

    def on_fit_start(self):
        self.metric_manager.set_device(self.device)
        self.test_metric_manager.set_device(self.device)

    def on_test_start(self):
        self.metric_manager.set_device(self.device)
        self.test_metric_manager.set_device(self.device)

    def training_step(self, batch, batch_step):
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        loss = self.loss.compute_loss(network_output_for_loss, type_representations)
        return loss

    def training_epoch_end(self, out):
        losses = [v for elem in out for k, v in elem.items()]
        self.logger_module.log_loss(name = 'train_loss', value = torch.mean(torch.tensor(losses)))

    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        network_output_for_inference = self.get_output_for_inference(network_output)
        loss = self.loss.compute_loss(network_output_for_loss, type_representations)

        inferred_types = self.inference_manager.infer_types(network_output_for_inference)
        
        if self.global_step > 0 or not self.avoid_sanity_logging:
            self.metric_manager.update(inferred_types, true_types)

        return loss
    
    def validation_epoch_end(self, out):
        if self.global_step > 0 or not self.avoid_sanity_logging:
            
            self.logger_module.add(key = 'epoch', value = self.current_epoch)

            metrics = self.metric_manager.compute()
            
            self.logger_module.log_all_metrics(metrics)
            val_loss = torch.mean(torch.tensor(out))
            self.logger_module.log_loss(name = 'val_loss', value = val_loss)
            self.logger_module.log_all()
            
            self.log("val_loss", val_loss)

    def test_step(self, batch, batch_step):
        _, _, true_types = batch
        network_output, _ = self.ET_Network(batch)
        network_output_for_inference = self.get_output_for_inference(network_output)
        inferred_types = self.inference_manager.infer_types(network_output_for_inference)
        self.test_metric_manager.update(inferred_types, true_types)
    
    def test_epoch_end(self, out):
        metrics = self.test_metric_manager.compute()
        metrics = { k: v.item() for k,v in metrics.items()}
        self.logger_module.log_all_metrics(metrics)
        self.logger_module.log_all()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def load_ET_Network(self, ET_Network_params, checkpoint_to_load):
        return IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number = self.type_number,
                                                                    type2id = self.type2id).load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, 
                                                                                                                            strict = False)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['state_dict'] = self.ET_Network.get_state_dict(smart_save=self.smart_save)
        del checkpoint['hyper_parameters']['logger']
        return super().on_save_checkpoint(checkpoint)

    def get_output_for_loss(self, network_output):
        return network_output
    
    def get_output_for_inference(self, network_output):
        return network_output

class IncrementalMainModule(MainModule):

    # def __init__(self, ET_Network_params: dict, type_number: int, type2id: dict, logger, loss_params, checkpoint_to_load: str = None, new_type_number = None):
    def __init__(self, ET_Network_params, type_number, type2id, inference_params, **kwargs):
        super().__init__(ET_Network_params=ET_Network_params, type_number=type_number, type2id=type2id, inference_params=inference_params, **kwargs)
        
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='')
        self.pretraining_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='pretraining')
        self.incremental_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='incremental')

        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test')
        self.test_pretraining_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_pretraining')
        self.test_incremental_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_incremental')


    def on_fit_start(self):
        self.metric_manager.set_device(self.device)
        self.pretraining_metric_manager.set_device(self.device)
        self.incremental_metric_manager.set_device(self.device)
        
        self.test_metric_manager.set_device(self.device)
        self.test_pretraining_metric_manager.set_device(self.device)
        self.test_incremental_metric_manager.set_device(self.device)

    def load_ET_Network(self, ET_Network_params, checkpoint_to_load):
        state_dict = torch.load(checkpoint_to_load)
        # get number of pretraining types
        checkpoint_type_number = state_dict['hyper_parameters']['type_number'] 
        # load ET_Network checkpoint
        checkpoint = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number=checkpoint_type_number,
                                                                    type2id=self.type2id).load_from_checkpoint(checkpoint_to_load = checkpoint_to_load,
                                                                                                                            strict = False)
        
        checkpoint.copy_pretrained_parameters_into_incremental_modules()
        checkpoint.freeze_pretrained_modules()

        return checkpoint

    def load_ET_Network_for_test(self, ET_Network_params, checkpoint_to_load):
        incremental_checkpoint_state_dict = torch.load(checkpoint_to_load)
        checkpoint = self.load_ET_Network(ET_Network_params=ET_Network_params, checkpoint_to_load=incremental_checkpoint_state_dict['hyper_parameters']['checkpoint_to_load'])
        checkpoint.load_from_checkpoint(checkpoint_to_load, strict = False)
        return checkpoint

    # NOTE: method to use externally without instantiating the class
    def load_ET_Network_for_test_(checkpoint_to_load):
        incremental_ckpt_state_dict = torch.load(checkpoint_to_load)
        pretrained_ckpt_state_dict = torch.load(incremental_ckpt_state_dict['hyper_parameters']['checkpoint_to_load'])
        ET_Network_params = incremental_ckpt_state_dict['hyper_parameters']['ET_Network_params']
        type_number = pretrained_ckpt_state_dict['hyper_parameters']['type_number']
        type2id = incremental_ckpt_state_dict['hyper_parameters']['type2id']
        ckpt = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number=type_number,
                                                                    type2id=type2id)
        ckpt.load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, strict = False)
        
        return ckpt

    def training_step(self, batch, batch_step):
        pretraining_batch = batch.pop('pretraining')
        incremental_batches = batch

        pretraining_loss = 0
        incremental_loss = 0
        
        keys = ['pretraining'] + list(incremental_batches.keys())
        batches = [pretraining_batch] + list(incremental_batches.values())

        for name, minibatch in zip(keys, batches): 
            network_output, type_representations = self.ET_Network(minibatch)
            network_output_for_loss = self.get_output_for_loss(network_output)
            loss = self.loss.compute_loss(network_output_for_loss, type_representations)
            if name == 'pretraining':
                pretraining_loss += loss
            else:
                incremental_loss += loss

        loss = pretraining_loss + incremental_loss
        return {'loss': loss, 'pretraining_loss': pretraining_loss, 'incremental_loss': incremental_loss}

    def training_epoch_end(self, out):
        # unpack losses
        average_losses = torch.mean(torch.tensor([list(o.values()) for o in out]), axis = 0)
        average_loss, average_pretraining_loss, average_incremental_loss = average_losses
        
        # wandb log
        self.logger_module.log_loss(name = 'losses/loss', value = average_loss)
        self.logger_module.log_loss(name = 'losses/pretraining_loss', value = average_pretraining_loss)
        self.logger_module.log_loss(name = 'losses/incremental_loss', value = average_incremental_loss)
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pretraining_val_loss = 0
        incremental_val_loss = 0

        _, _, true_types = batch
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        network_output_for_inference = self.get_output_for_inference(network_output)
        loss = self.loss.compute_loss(network_output_for_loss, type_representations)
        inferred_types = self.inference_manager.infer_types(*network_output_for_inference)

        if self.global_step > 0 or not self.avoid_sanity_logging:
        # collect predictions for all val_dataloaders
            self.metric_manager.update(inferred_types, true_types)
        
        if dataloader_idx == 0: # batches from pretraining val dataloader
            if self.global_step > 0 or not self.avoid_sanity_logging:
                # collect predictions for pretraining val_dataloaders
                self.pretraining_metric_manager.update(inferred_types, true_types)
                pretraining_val_loss += loss
        else: # batches from incremental val dataloaders
            if self.global_step > 0 or not self.avoid_sanity_logging:
                # collect predictions for incremental val_dataloaders
                self.incremental_metric_manager.update(inferred_types, true_types)
                incremental_val_loss += loss
        
        val_loss = pretraining_val_loss + incremental_val_loss   

        return val_loss, pretraining_val_loss, incremental_val_loss
    
    def validation_epoch_end(self, out):
        # compose the returned losses to the expected shape [pretraining_batches + incremental_batches, len(self.validation_step())] 
        dataloader0_losses = torch.tensor(out[0])
        dataloader1_losses = torch.tensor(out[1])
        out = torch.concat([dataloader0_losses, dataloader1_losses], dim = 0)

        # compute the average of losses ignoring zeros (zeros are inserted by Lightning)
        mask = out != 0
        average_losses = (out*mask).sum(dim=0)/mask.sum(dim=0)
        
        average_val_loss, average_pretraining_val_loss, average_incremental_val_loss = average_losses
            
        if self.global_step > 0 or not self.avoid_sanity_logging:
            # wandb log
            self.logger_module.add(key = 'epoch', value = self.current_epoch)

            metrics = self.metric_manager.compute()
            self.logger_module.log_all_metrics(metrics)

            pretraining_metrics = self.pretraining_metric_manager.compute()
            self.logger_module.log_all_metrics(pretraining_metrics)

            incremental_metrics = self.incremental_metric_manager.compute()
            self.logger_module.log_all_metrics(incremental_metrics)

            self.logger_module.log_loss(name = 'losses/val_loss', value = average_val_loss)
            self.logger_module.log_loss(name = 'losses/pretraining_val_loss', value = average_pretraining_val_loss)
            self.logger_module.log_loss(name = 'losses/incremental_val_loss', value = average_incremental_val_loss)
            
            self.logger_module.log_all()

            # callback log
            self.log("losses/val_loss", average_val_loss)
            self.log("losses/pretraining_val_loss", average_pretraining_val_loss)
            self.log("losses/incremental_val_loss", average_incremental_val_loss)
    
    # # TODO: ???
    # def test_step(self, batch, batch_idx, dataloader_idx):
    #     _, _, true_types = batch
    #     network_output, _ = self.ET_Network(batch)
    #     network_output_for_inference = self.get_output_for_inference(network_output)
    #     inferred_types = self.inference_manager.infer_types(network_output_for_inference)

    #     # collect predictions for all test_dataloaders
    #     self.test_metric_manager.update(inferred_types, true_types)
        
    #     if dataloader_idx == 0: # batches from pretraining test dataloader
    #         # collect predictions for pretraining test_dataloaders
    #         self.test_pretraining_metric_manager.update(inferred_types, true_types)
    #     else: # batches from incremental test dataloaders
    #         # collect predictions for incremental test_dataloaders
    #         self.test_incremental_metric_manager.update(inferred_types, true_types)
    
    def test_epoch_end(self, out):
        # wandb log
        test_metrics = self.test_metric_manager.compute()
        self.logger_module.log_all_metrics(test_metrics)

        test_pretraining_metrics = self.test_pretraining_metric_manager.compute()
        self.logger_module.log_all_metrics(test_pretraining_metrics)

        test_incremental_metrics = self.test_incremental_metric_manager.compute()
        self.logger_module.log_all_metrics(test_incremental_metrics)

        self.logger_module.log_all()
    
    def get_output_for_loss(self, network_output):
        return torch.concat(network_output, dim=1)
        
class KENNMainModule(MainModule):
    def get_output_for_inference(self, network_output):
        # return postkenn output
        return network_output[1]

    def get_output_for_loss(self, network_output):
        # return postkenn output
        return network_output[1]

class KENNMultilossMainModule(KENNMainModule):
    def get_output_for_loss(self, network_output):
        # return prekenn and postkenn output (same as returning the output as is...)
        return network_output[0], network_output[1]

class IncrementalKENNMainModule(KENNMainModule, IncrementalMainModule):
    # NOTE: depth-first left-to-right MRO, do not change inheritance order!
    
    def get_output_for_loss(self, network_output):
        # return postkenn output
        return torch.concat(network_output[1], dim=1)
    

class IncrementalKENNMultilossMainModule(KENNMultilossMainModule, IncrementalMainModule):
    # NOTE: depth-first left-to-right MRO, do not change inheritance order!
    
    def get_output_for_loss(self, network_output):
        # return prekenn and postkenn output (same as returning the output as is...)
        return torch.concat(network_output[0], dim=1), torch.concat(network_output[1], dim=1)

class BoxEmbeddingMainModule(MainModule):
    def get_output_for_inference(self, network_output):
        return torch.exp(network_output[1])
    
    def get_output_for_loss(self, network_output):
        # return log probs
        return network_output[1]

class IncrementalBoxEmbeddingMainModule(BoxEmbeddingMainModule, IncrementalMainModule):
    # this is strange: BoxEmbeddingIncrementalProjector doesn't have a forward, so it inherits the forward from IncrementalMainModule
    # the forward of IncrementalMainModule uses BoxEmbeddingProjector.classify() and BoxEmbeddingProjector.project_input()
    # that differently from BoxEmbeddingProjector.classify() return the logits. 
    # 
    # So this is the cause of different get_output_for_inference and get_output_for_loss methods 
    # between BoxEmbeddingMainModule and IncrementalBoxEmbeddingMainModule
    def get_output_for_inference(self, network_output):
        return torch.exp(network_output[0]), torch.exp(network_output[1])
    
    def get_output_for_loss(self, network_output):
        # return log probs by concatenating pretrained and incremental outputs
        return torch.concat(network_output, dim=1)
