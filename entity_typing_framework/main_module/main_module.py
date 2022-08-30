from tabnanny import check
from typing import Any, Dict
# from entity_typing_framework.main_module.inference_manager import BaseInferenceManager
from entity_typing_framework.main_module.metric_manager import MetricManager, MetricManagerForIncrementalTypes
from entity_typing_framework.utils.implemented_classes_lvl0 import IMPLEMENTED_CLASSES_LVL0
from pytorch_lightning.core.lightning import LightningModule
import torch
import time
import numpy as np

class MainModule(LightningModule):
    def __init__(self, 
    ET_Network_params : dict,
    type_number : int,
    type2id : dict,
    logger,
    loss_module_params,
    inference_params : dict,
    checkpoint_to_load : str = None,
    avoid_sanity_logging : bool = False,
    smart_save : bool = True,
    learning_rate : float = 5e-4 # the default value is the one used in EMNLP 2022 experiments
    ):

        super().__init__()
        self.type_number = type_number
        self.logger_module = logger
        self.ET_Network_params = ET_Network_params
        self.type2id = type2id
        self.avoid_sanity_logging = avoid_sanity_logging
        self.smart_save = smart_save
        self.learning_rate = learning_rate

        if not checkpoint_to_load:
            self.ET_Network = IMPLEMENTED_CLASSES_LVL0[self.ET_Network_params['name']](**self.ET_Network_params, type_number = self.type_number, type2id = self.type2id)
        else:
            self.ET_Network = self.load_ET_Network(ET_Network_params=self.ET_Network_params, checkpoint_to_load=checkpoint_to_load)
        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='dev')
        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test')

        self.inference_manager = IMPLEMENTED_CLASSES_LVL0[inference_params['name']](type2id=self.type2id, **inference_params)
        self.loss = IMPLEMENTED_CLASSES_LVL0[loss_module_params['name']](type2id=self.type2id, **loss_module_params)
        self.save_hyperparameters()
        self.log_validation_metrics = True # this flag will be disabled during threshold calibration

    def on_fit_start(self):
        self.metric_manager.set_device(self.device)
        self.test_metric_manager.set_device(self.device)

    def on_test_start(self):
        # set devices
        self.metric_manager.set_device(self.device)
        self.test_metric_manager.set_device(self.device)
    #     # calibrate threshold if requested
    #     if self.inference_manager.calibrate_threshold:
    #         # disable validation metrics flag
    #         self.log_validation_metrics = False
    #         # iterate over thresholds and call validation routine
    #         max_f1 = 0
    #         max_t = 0
    #         for t in np.arange(.05, 1, .5): # TODO: change to .05
    #             self.inference_manager.threshold = t
    #             self.trainer.validate(self.trainer.model, self.trainer.datamodule.val_dataloader())
    #             f1 = self.last_validation_metrics['dev/macro_example/f1']
    #             if f1 > max_f1:
    #                 max_f1 = f1
    #                 max_t = t
            
    #         # set optimal threshold
    #         self.inference_manager.threshold = max_t

    def training_step(self, batch, batch_step):
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        loss = self.loss.compute_loss_for_training_step(encoded_input=network_output_for_loss, type_representation=type_representations)
        return loss

    def training_epoch_end(self, out):
        losses = [v for elem in out for k, v in elem.items()]
        self.logger_module.log_loss(name = 'train_loss', value = torch.mean(torch.tensor(losses)))

    def validation_step(self, batch, batch_step):
        _, _, true_types = batch
        true_types = self.get_true_types_for_metrics(true_types)
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        network_output_for_inference = self.get_output_for_inference(network_output)

        # TODO: check if is the same for every projector
        loss = self.loss.compute_loss_for_validation_step(encoded_input=network_output_for_loss, type_representation=true_types)
        # loss = self.loss.compute_loss_for_validation_step(network_output_for_loss, type_representations)

        inferred_types = self.inference_manager.infer_types(network_output_for_inference)
        
        if self.global_step > 0 or not self.avoid_sanity_logging:
            self.metric_manager.update(inferred_types, true_types)

        return loss
    
    def validation_epoch_end(self, out):
        if self.global_step > 0 or not self.avoid_sanity_logging:
            

            metrics = self.metric_manager.compute()
            
            if self.log_validation_metrics:
                self.logger_module.add(key = 'epoch', value = self.current_epoch)
                self.logger_module.log_all_metrics(metrics)
                val_loss = torch.mean(torch.tensor(out))
                self.logger_module.log_loss(name = 'val_loss', value = val_loss)
                self.logger_module.log_all()
                self.log("val_loss", val_loss)
            
            # used for threshold calibration
            self.last_validation_metrics = metrics
                

    def test_step(self, batch, batch_step):
        _, _, true_types = batch
        true_types = self.get_true_types_for_metrics(true_types)
        network_output, _ = self.ET_Network(batch)
        network_output_for_inference = self.get_output_for_inference(network_output)
        inferred_types = self.inference_manager.infer_types(network_output_for_inference)
        self.test_metric_manager.update(inferred_types, true_types)
    
    def test_epoch_end(self, out):
        metrics = self.test_metric_manager.compute()
        metrics = { k: v.item() for k,v in metrics.items()}
        metrics['threshold'] = self.inference_manager.threshold
        self.logger_module.log_all_metrics(metrics)
        self.logger_module.log_all()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def load_ET_Network(self, ET_Network_params, checkpoint_to_load):
        return IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number = self.type_number,
                                                                    type2id = self.type2id).load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, 
                                                                                                                            strict = False)

    def load_ET_Network_for_test_(checkpoint_to_load):

        print('Loading model with torch.load ...')
        start = time.time()
        ckpt_state_dict = torch.load(checkpoint_to_load)
        print('Loaded in {:.2f} seconds'.format(time.time() - start))
        
        ET_Network_params = ckpt_state_dict['hyper_parameters']['ET_Network_params']
        type_number = ckpt_state_dict['hyper_parameters']['type_number']
        type2id = ckpt_state_dict['hyper_parameters']['type2id']
        
        print('Instantiating {} class ...'.format(ET_Network_params['name']))
        start = time.time()
        ckpt = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number=type_number,
                                                                    type2id=type2id)
        print('Instantiated in {:.2f} seconds'.format(time.time() - start))
        
        print('Loading from checkpoint with load_from_checkpoint...')
        start = time.time()
        ckpt.load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, strict = False)
        print('Loaded in {:.2f} seconds'.format(time.time() - start))
        
        return ckpt

    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['state_dict'] = self.ET_Network.get_state_dict(smart_save=self.smart_save)
        del checkpoint['hyper_parameters']['logger']
        return super().on_save_checkpoint(checkpoint)

    def get_output_for_loss(self, network_output):
        return network_output
    
    def get_output_for_inference(self, network_output):
        return network_output
    
    def get_true_types_for_metrics(self, true_types):
        return self.inference_manager.transform_true_types(true_types)

# TODO: add threshold calibration
class IncrementalMainModule(MainModule):

    # def __init__(self, ET_Network_params: dict, type_number: int, type2id: dict, logger, loss_params, checkpoint_to_load: str = None, new_type_number = None):
    def __init__(self, ET_Network_params, type_number, test_index, type2id, inference_params, **kwargs):
        super().__init__(ET_Network_params=ET_Network_params, type_number=type_number, type2id=type2id, inference_params=inference_params, **kwargs)
        
        # test_index is always equal to the number of incremental types + 1 since we have a dev set for each incremental type
        self.test_index = test_index

        ### prepare type2ids for incremental metrics
        # filter incremental types
        type2id_pretrained = self.ET_Network.input_projector.pretrained_projector.type2id
        type2id_all = self.ET_Network.input_projector.additional_projector.type2id
        types_incremental = set(type2id_all.keys()) - set(type2id_pretrained.keys())
        type2id_incremental = { type_incremental : type2id_all[type_incremental] for type_incremental in types_incremental }
        self.type2id_incremental = type2id_incremental
        # add fathers of incremental types
        fathers = set()
        for t in type2id_incremental.keys():
            f = t.replace(f"/{t.split('/')[-1]}", "")
            fathers.add(f)
        
        fathers = list(fathers)
        self.type2id_evaluation = { f : type2id_all[f] for f in fathers }
        self.type2id_evaluation.update(type2id_incremental)

        self.metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='')
        self.pretraining_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='pretraining')
        self.incremental_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='incremental')

        self.test_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_evolution')
        self.test_pretraining_metric_manager = MetricManager(num_classes=self.type_number, device=self.device, prefix='test_pretraining_evolution')
        self.test_incremental_metric_manager = MetricManager(num_classes=len(self.type2id_evaluation), device=self.device, prefix='test_incremental_evolution')
        self.test_incremental_only_metric_manager = MetricManager(num_classes=len(self.type2id_incremental), device=self.device, prefix='test_incremental_only_evolution')
        
        self.test_incremental_specific_metric_manager = MetricManagerForIncrementalTypes(self.type_number, device=self.device, prefix='test_incremental_evolution')

        self.test_incremental_exclusive_metric_manager = MetricManager(len(self.type2id_evaluation), device=self.device, prefix='test_incremental_evolution')
        self.test_incremental_only_exclusive_metric_manager = MetricManager(len(self.type2id_incremental), device=self.device, prefix='test_incremental_only_evolution')


    def on_fit_start(self):
        self.metric_manager.set_device(self.device)
        self.pretraining_metric_manager.set_device(self.device)
        self.incremental_metric_manager.set_device(self.device)
        
        self.test_metric_manager.set_device(self.device)
        self.test_pretraining_metric_manager.set_device(self.device)
        self.test_incremental_metric_manager.set_device(self.device)
        self.test_incremental_only_metric_manager.set_device(self.device)

        self.test_incremental_specific_metric_manager.set_device(self.device)
        self.test_incremental_exclusive_metric_manager.set_device(self.device)
        self.test_incremental_only_exclusive_metric_manager.set_device(self.device)

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
        print('Loading incremental model with torch.load ...')
        start = time.time()
        incremental_ckpt_state_dict = torch.load(checkpoint_to_load)
        print('Loaded in {:.2f} seconds'.format(time.time() - start))
        
        start = time.time()
        print('Loading pretrained model with torch.load ...')
        pretrained_ckpt_state_dict = torch.load(incremental_ckpt_state_dict['hyper_parameters']['checkpoint_to_load'])
        print('Loaded in {:.2f} seconds'.format(time.time() - start))
        
        ET_Network_params = incremental_ckpt_state_dict['hyper_parameters']['ET_Network_params']
        type_number = pretrained_ckpt_state_dict['hyper_parameters']['type_number']
        type2id = incremental_ckpt_state_dict['hyper_parameters']['type2id']
        print('Instantiating {} class ...'.format(ET_Network_params['name']))
        start = time.time()
        ckpt = IMPLEMENTED_CLASSES_LVL0[ET_Network_params['name']](**ET_Network_params, 
                                                                    type_number=type_number,
                                                                    type2id=type2id)
        print('Instantiated in {:.2f} seconds'.format(time.time() - start))
        
        print('Loading from checkpoint with load_from_checkpoint...')
        start = time.time()
        ckpt.load_from_checkpoint(checkpoint_to_load = checkpoint_to_load, strict = False)
        print('Loaded in {:.2f} seconds'.format(time.time() - start))
        
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
            loss = self.loss.compute_loss_for_training_step(encoded_input=network_output_for_loss, type_representation=type_representations)
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
        true_types = self.get_true_types_for_metrics(true_types)
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_loss = self.get_output_for_loss(network_output)
        network_output_for_inference = self.get_output_for_inference(network_output)
        loss = self.loss.compute_loss_for_validation_step(encoded_input=network_output_for_loss, type_representation=type_representations)
        inferred_types = self.inference_manager.infer_types(network_output_for_inference)

        if self.global_step > 0 or not self.avoid_sanity_logging:
        # collect predictions for all val_dataloaders
            self.metric_manager.update(inferred_types, true_types)
        
        if self.global_step > 0 or not self.avoid_sanity_logging:
            if dataloader_idx == 0: # batches from pretraining val dataloader
                # collect predictions for pretraining val_dataloaders
                self.pretraining_metric_manager.update(inferred_types, true_types)
                pretraining_val_loss += loss
            elif dataloader_idx == self.test_index:

                # collect predictions for incremental test_dataloaders
                self.test_metric_manager.update(inferred_types, true_types)
                self.test_pretraining_metric_manager.update(inferred_types, true_types)
                self.test_incremental_specific_metric_manager.update(inferred_types, true_types)

                # NOTE: exclude fathers from aggregation
                # prepare filtered predictions with only incremental types
                idx = torch.tensor(list(self.type2id_incremental.values()))
                y_true_filtered = true_types.index_select(dim=1, index=idx.cuda())
                inferred_types_filtered = inferred_types.index_select(dim=1, index=idx.cuda())
                self.test_incremental_only_metric_manager.update(inferred_types_filtered.cuda(), y_true_filtered.cuda())
                # prepare filtered predictions with only incremental types and using only the examples that has at least one of the types of interes as true type 
                idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
                y_true_filtered = y_true_filtered.index_select(dim=0, index=idx.cuda())
                if torch.sum(y_true_filtered):
                    inferred_types_filtered = inferred_types_filtered.index_select(dim=0, index=idx.cuda())
                    self.test_incremental_only_exclusive_metric_manager.update(inferred_types_filtered, y_true_filtered)

                # NOTE: include fathers in aggregation
                # prepare filtered predictions with only incremental types and their fathers
                idx = torch.tensor(list(self.type2id_evaluation.values()))
                y_true_filtered = true_types.index_select(dim=1, index=idx.cuda())
                inferred_types_filtered = inferred_types.index_select(dim=1, index=idx.cuda())
                self.test_incremental_metric_manager.update(inferred_types_filtered, y_true_filtered)
                # prepare filtered predictions with only incremental types and their fathers and using only the examples that has at least one of the types of interes as true type 
                idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
                y_true_filtered = y_true_filtered.index_select(dim=0, index=idx.cuda())
                if torch.sum(y_true_filtered):
                    inferred_types_filtered = inferred_types_filtered.index_select(dim=0, index=idx.cuda())
                    self.test_incremental_exclusive_metric_manager.update(inferred_types_filtered, y_true_filtered)


            else: # batches from incremental val dataloaders
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
            
            self.log_test_metrics()

            self.logger_module.log_all()

            # callback log
            self.log("losses/val_loss", average_val_loss)
            self.log("losses/pretraining_val_loss", average_pretraining_val_loss)
            self.log("losses/incremental_val_loss", average_incremental_val_loss)
    
    def test_step(self, batch, batch_idx):
        _, _, true_types = batch
        true_types = self.get_true_types_for_metrics(true_types)
        network_output, type_representations = self.ET_Network(batch)
        network_output_for_inference = self.get_output_for_inference(network_output)
        inferred_types = self.inference_manager.infer_types(*network_output_for_inference)

        # collect predictions for incremental test_dataloaders
        self.test_metric_manager.update(inferred_types, true_types)
        self.test_pretraining_metric_manager.update(inferred_types, true_types)
        self.test_incremental_specific_metric_manager.update(inferred_types, true_types)

        # NOTE: exclude fathers from aggregation
        # prepare filtered predictions with only incremental types
        idx = torch.tensor(list(self.type2id_incremental.values()))
        y_true_filtered = true_types.index_select(dim=1, index=idx.cuda())
        inferred_types_filtered = inferred_types.index_select(dim=1, index=idx.cuda())
        self.test_incremental_only_metric_manager.update(inferred_types_filtered.cuda(), y_true_filtered.cuda())
        # prepare filtered predictions with only incremental types and using only the examples that has at least one of the types of interes as true type 
        idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
        y_true_filtered = y_true_filtered.index_select(dim=0, index=idx.cuda())
        if torch.sum(y_true_filtered):
            inferred_types_filtered = inferred_types_filtered.index_select(dim=0, index=idx.cuda())
            self.test_incremental_only_exclusive_metric_manager.update(inferred_types_filtered, y_true_filtered)

        # NOTE: include fathers in aggregation
        # prepare filtered predictions with only incremental types and their fathers
        idx = torch.tensor(list(self.type2id_evaluation.values()))
        y_true_filtered = true_types.index_select(dim=1, index=idx.cuda())
        inferred_types_filtered = inferred_types.index_select(dim=1, index=idx.cuda())
        self.test_incremental_metric_manager.update(inferred_types_filtered, y_true_filtered)
        # prepare filtered predictions with only incremental types and their fathers and using only the examples that has at least one of the types of interes as true type 
        idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
        y_true_filtered = y_true_filtered.index_select(dim=0, index=idx.cuda())
        if torch.sum(y_true_filtered):
            inferred_types_filtered = inferred_types_filtered.index_select(dim=0, index=idx.cuda())
            self.test_incremental_exclusive_metric_manager.update(inferred_types_filtered, y_true_filtered)


    
    def test_epoch_end(self, out):
        self.log_test_metrics()

    def log_test_metrics(self):
        test_metrics = self.test_metric_manager.compute()
        self.logger_module.log_all_metrics(test_metrics)

        test_pretraining_metrics = self.test_pretraining_metric_manager.compute()
        self.logger_module.log_all_metrics(test_pretraining_metrics)

        test_incremental_metrics = self.test_incremental_metric_manager.compute()
        self.logger_module.log_all_metrics(test_incremental_metrics)

        test_incremental_only_metrics = self.test_incremental_only_metric_manager.compute()
        self.logger_module.log_all_metrics(test_incremental_only_metrics)
        
        test_incremental_specific_metrics = self.test_incremental_specific_metric_manager.compute(self.type2id_evaluation)
        self.logger_module.log_all_metrics(test_incremental_specific_metrics)
                    
        test_incremental_exclusive_metrics = self.test_incremental_exclusive_metric_manager.compute()
        metrics_test_incremental_aggregated = {k.replace('macro_example', 'macro_example_exclusive'): v.item() for k,v in test_incremental_exclusive_metrics.items() if 'macro_example' in k}
        self.logger_module.log_all_metrics(metrics_test_incremental_aggregated)

        test_incremental_only_exclusive_metrics = self.test_incremental_only_exclusive_metric_manager.compute()
        metrics_test_incremental_only_exclusive_aggregated = {k.replace('macro_example', 'macro_example_exclusive'): v.item() for k,v in test_incremental_only_exclusive_metrics.items() if 'macro_example' in k}
        self.logger_module.log_all_metrics(metrics_test_incremental_only_exclusive_aggregated)
    
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

class ClassifierNFETCMainModule(MainModule):
    def __init__(self, ET_Network_params: dict, type_number: int, type2id: dict, logger, loss_module_params, inference_params: dict, checkpoint_to_load: str = None, avoid_sanity_logging: bool = False, smart_save: bool = True):
        super().__init__(ET_Network_params, type_number, type2id, logger, loss_module_params, inference_params, checkpoint_to_load, avoid_sanity_logging, smart_save)
        self.loss.create_prior(self.type2id)

class BoxKENNMultilossMainModule(KENNMultilossMainModule):
    def get_output_for_inference(self, network_output):
        return torch.exp(super().get_output_for_inference(network_output))

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

class BoxEmbeddingMainModuleForOpt(BoxEmbeddingMainModule):
    # def test_step(self, batch, batch_step):
    #     pass

    def on_fit_end(self) -> None:
        with open('opt_losses.txt', 'a') as out:
            out.write(f'{self.trainer.checkpoint_callback.best_model_score.item()}\n')