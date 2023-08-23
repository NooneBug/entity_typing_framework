from pytorch_lightning.core.lightning import LightningModule
from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
import torch 
from copy import deepcopy

class BaseEntityTypingNetwork(LightningModule):
    '''
    Basic :ref:`EntityTypingNetwork <EntityTypingNetwork>`. This module is able to use the following submodules:

    :ref:`encoder <encoder>`:
        :ref:`entity_typing_framework.EntityTypingNetwork_classes.input_encoders.DistilBERTEncoder <DistilBERTEncoder>`

        :ref:`entity_typing_framework.EntityTypingNetwork_classes.input_encoders.AdapterDistilBERTEncoder <AdapterDistilBERTEncoder>`

    :ref:`type_encoder <type_encoder>`:
        :ref:`entity_typing_framework.EntityTypingNetwork_classes.type_encoders.OneHotTypeEncoder <OneHotTypeEncoder>`

    :ref:`input_projector <input_projector>`:
        :ref:`entity_typing_framework.EntityTypingNetwork_classes.type_encoders.Classifier <Classifier>`


    Parameters:
        name:
            the name of the module, specified in the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.name`. Has to be declared in the :doc:`module_dictionary`
        
        network_params:
            parameters for the module and for the submodules, specified in the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.network_params`

            expected keys in network_params are: :code:`model.ET_Network_params.network_params.encoder_params`, :code:`model.ET_Network_params.network_params.type_encoder_params`, and :code:`model.ET_Network_params.network_params.input_projector_params`

        type_number:
            number of types for this run. Automatic managed through :ref:`DatasetManager <DatasetManager>`
    '''
    def __init__(self, name, network_params, type_number, type2id
    # , encoder_params, type_encoder_params, 
    # inference_params, metric_manager_params, loss_params, 
    ):
        super().__init__()

        self.type2id = type2id
        self.type_number = type_number
        
        self.instance_encoder(network_params)

        self.instance_input_projector(network_params)

        self.instance_type_encoder(network_params)


    def instance_encoder(self, network_params):
        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES_LVL1[encoder_params['name']](**encoder_params)

    def instance_type_encoder(self, network_params):
        type_encoder_params = network_params['type_encoder_params']
        self.type_encoder = IMPLEMENTED_CLASSES_LVL1[type_encoder_params['name']](num_embeddings=self.type_number, **type_encoder_params)

    def instance_input_projector(self, network_params):
        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=self.type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
                                            type2id = self.type2id,
                                            **input_projector_params)

    def forward(self, batch):
        '''
        override of :code:pytorch_lightning.LightningModule.forward (`ref <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_)

        parameters:
            batch:
                the batch returned by the :ref:`Dataset <dataset>`
        
        return:
            projected_input:
                output of the :ref:`input_projector <input_projector>`. Commonly the :ref:`input_projector <input_projector>` takes in input the output of the :ref:`encoder <encoder>`
            
            encoded_types:
                output of the :ref:`type_encoder <type_encoder>`.
        '''
        batched_sentences, batched_attn_masks, batched_labels = batch
        
        encoded_input = self.encoder(batched_sentences, batched_attn_masks)
        projected_input = self.input_projector(encoded_input)
        encoded_types = self.type_encoder(batched_labels)
        
        return projected_input, encoded_types

    def load_from_checkpoint(self, checkpoint_to_load, strict):
        state_dict = torch.load(checkpoint_to_load)
        s_dict = deepcopy(state_dict['state_dict'])
        renamed_state_dict = self.get_renamed_state_dict(s_dict)
        model_state_dict = self.state_dict()
        is_changed = False
        for k in renamed_state_dict:
            if k in model_state_dict:
                if renamed_state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {renamed_state_dict[k].shape}")
                    old_class_number = renamed_state_dict[k].shape[0]
                    if len(renamed_state_dict[k].shape) == 2:
                        model_state_dict[k][:old_class_number, :] = renamed_state_dict[k]
                    elif len(renamed_state_dict[k].shape) == 1:
                        model_state_dict[k][:old_class_number] = renamed_state_dict[k]

                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            state_dict.pop("optimizer_states", None)

        self.load_state_dict(renamed_state_dict, strict=strict)
        return self

    def get_renamed_state_dict(self, state_dict):
        renamed_state_dict = {k.replace('ET_Network.', ''): v for k, v in state_dict.items()}

        return renamed_state_dict
    
    def get_state_dict(self, smart_save=True):
        state_dict = {}
        state_dict.update(self.add_prefix_to_dict(self.encoder.get_state_dict(smart_save), prefix='encoder'))
        state_dict.update(self.add_prefix_to_dict(self.type_encoder.get_state_dict(smart_save), prefix='type_encoder'))
        state_dict.update(self.add_prefix_to_dict(self.input_projector.get_state_dict(smart_save), prefix='input_projector'))
        return state_dict
    
    def add_prefix_to_dict(self, state_dict, prefix):
        if state_dict:
            return {f'{prefix}.{k}': v for k, v in state_dict.items()}
        else:
            return {}        

class IncrementalEntityTypingNetwork(BaseEntityTypingNetwork):
    def get_renamed_state_dict(self, state_dict):
        '''
        This method is called in two use cases:
            1) loading a pretrained network to setup incremental training
            2) loading an incremental network to test it
        '''

        parameters = list(state_dict.keys())

        if any('pretrained_projector' in p for p in parameters):
            # loading an incremental network to test it
            # keep the state dict as is
            renamed_state_dict = state_dict
        else:
            # loading a pretrained network to setup incremental training
            father_renamed_state_dict = super().get_renamed_state_dict(state_dict)
            #rename the father's input_projector params into input_projector.pretrained_projector.params
            renamed_state_dict = {k.replace('input_projector', 'input_projector.pretrained_projector'): v for k, v in father_renamed_state_dict.items() if 'pretrained_projector' not in k and 'additional_projector' not in k }

        return renamed_state_dict
    
    def copy_pretrained_parameters_into_incremental_modules(self):
        self.input_projector.copy_pretrained_parameters_into_incremental_module()
        
        # for now (before EMNLP2022 deadline) these are not useful since encoder will be frozen and type_encoder is a dummy module (no parameters) 
        # self.encoder.copy_pretrained_parameters_into_incremental_module()
        # self.type_encoder.copy_pretrained_parameters_into_incremental_module()
    
    def freeze_pretrained_modules(self):
        self.encoder.freeze()
        self.input_projector.freeze_pretrained()

# class BoxEmbeddingEntityTypingNetwork(BaseEntityTypingNetwork):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def forward(self, batch, is_training = True):
#         #TODO: write the documentation
#         '''
#         override of :code:pytorch_lightning.LightningModule.forward (`ref <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_)

#         parameters:
#             batch:
#                 the batch returned by the :ref:`Dataset <dataset>`
        
#         return:
#             projected_input:
#                 output of the :ref:`input_projector <input_projector>`. Commonly the :ref:`input_projector <input_projector>` takes in input the output of the :ref:`encoder <encoder>`
            
#             encoded_types:
#                 output of the :ref:`type_encoder <type_encoder>`.
#         '''
#         batched_sentences, batched_attn_masks, batched_labels = batch
        
#         encoded_input = self.encoder(batched_sentences, batched_attn_masks)
#         projected_input, log_probs = self.input_projector(encoded_input)
#         encoded_types = self.type_encoder(batched_labels)
        
#         return log_probs, encoded_types

    
class CrossDatasetEntityTypingNetwork(BaseEntityTypingNetwork):
    def __init__(self, name, network_params, type_number, type2id):
        super().__init__(name, network_params, type_number, type2id)
        self.copy_pretrained_parameters_into_incremental_modules(network_params)
    
    def copy_pretrained_parameters_into_incremental_modules(self, network_params):
        # self.input_projector.copy_src_parameters_into_tgt_module()
        self.copy_encoder(network_params)
        # self.freeze_pretrained_modules()

        
    # TODO: it should not be in this class
    def copy_encoder(self, network_params):
        ckpt = torch.load(network_params['input_projector_params']['src_ckpt'])
        if any(['adapter' in k for k in ckpt.keys()]) and hasattr(self.encoder, 'reduction_factor'):
            encoder_state_dict = {'.'.join(k.split('.')[1:]):v for k,v in ckpt['state_dict'].items() if 'adapters' in k}
        else:
            encoder_state_dict = ckpt['state_dict']
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        

    def freeze_pretrained_modules(self):
        self.encoder.freeze()
        self.input_projector.freeze_src_classifier()

class ALIGNIENetwork(BaseEntityTypingNetwork):

    def __init__(self, name, network_params, type_number, type2id, mask_token_id, init_verbalizer, vocab_size):
        self.mask_token_id = mask_token_id
        self.init_verbalizer = init_verbalizer
        self.vocab_size = vocab_size
        super().__init__(name, network_params, type_number, type2id)

    def instance_encoder(self, network_params):
        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES_LVL1[encoder_params['name']](**encoder_params,
                                                                        mask_token_id = self.mask_token_id)

    def instance_input_projector(self, network_params):
        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=self.type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
                                            type2id = self.type2id,
                                            verbalizer = self.init_verbalizer,
                                            vocab_size = self.vocab_size,
                                            **input_projector_params)

    def update_verbalizer(self, epoch_id):
        return self.input_projector.update_verbalizer(epoch_id)
    
class PROMETNetwork(BaseEntityTypingNetwork):

    def __init__(self, name, network_params, type_number, type2id, mask_token_id):
        self.mask_token_id = mask_token_id
        super().__init__(name, network_params, type_number, type2id)

    def instance_encoder(self, network_params):
        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES_LVL1[encoder_params['name']](**encoder_params,
                                                                        mask_token_id = self.mask_token_id)

    def instance_input_projector(self, network_params):
        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=self.type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
                                            type2id = self.type2id,
                                            **input_projector_params)
        
    def forward(self, batch):

        batched_sentences, batched_attn_masks, batched_labels = batch
        
        encoded_input = self.encoder(batched_sentences, batched_attn_masks)
        projected_input = self.input_projector(encoded_input)
        encoded_types = self.type_encoder(batched_labels)

        return projected_input, encoded_types