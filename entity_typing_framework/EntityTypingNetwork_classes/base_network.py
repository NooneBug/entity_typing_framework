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
        renamed_state_dict = {k.replace('ET_Network.', ''): v for k, v in s_dict.items()}
        
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

class BoxEmbeddingEntityTypingNetwork(BaseEntityTypingNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def instance_type_encoder(self, network_params):

        box_embeddings_dimension = network_params['box_embeddings_dimension']
        type_encoder_params = network_params['type_encoder_params']
        self.type_encoder = IMPLEMENTED_CLASSES_LVL1[type_encoder_params['name']](num_embeddings=self.type_number, 
                                                                                    embedding_dim = box_embeddings_dimension,
                                                                                    **type_encoder_params)
    def instance_input_projector(self, network_params):
        box_embeddings_dimension = network_params['box_embeddings_dimension']
        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=self.type_number, 
                                            input_dim = self.encoder.get_representation_dim(),
                                            box_embeddings_dimension = box_embeddings_dimension,
                                            **input_projector_params)
    
    def forward(self, batch, is_training = True):
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
        log_probs, loss_weights, targets = self.type_encoder(mc_box = projected_input, 
                                            targets = batched_labels,
                                            is_training = is_training)
        
        return projected_input, log_probs, loss_weights, targets

    
