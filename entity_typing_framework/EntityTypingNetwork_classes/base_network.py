from pytorch_lightning.core.lightning import LightningModule
from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
import torch 

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

        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES_LVL1[encoder_params['name']](**encoder_params)

        type_encoder_params = network_params['type_encoder_params']
        self.type_encoder = IMPLEMENTED_CLASSES_LVL1[type_encoder_params['name']](type_number=type_number, **type_encoder_params)

        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=type_number, 
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

    def load_from_checkpoint(self, checkpoint_to_load, strict, **kwargs):
        state_dict = torch.load(checkpoint_to_load)
        renamed_state_dict = {k.replace('ET_Network.', ''): v for k, v in state_dict['state_dict'].items()}
        
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

        # self.load_state_dict(new_state_dict, strict=strict)
        return self


class EntityTypingNetworkForIncrementalTraining(BaseEntityTypingNetwork):
    def setup_incremental_training(self, new_type_number, network_params):
        input_projector_params = network_params['input_projector_params']

        ## extract last classifier layer and manually insert the out_features number
        single_layers = sorted(input_projector_params['layers_parameters'].items())
        single_layers[-1][1]['out_features'] = new_type_number
        input_projector_params['layers_parameters'] = {k: v for k, v in single_layers}
        
        self.freeze()

        self.additional_input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=new_type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
                                            **input_projector_params)
    
    def forward(self, batch):
        batched_sentences, batched_attn_masks, batched_labels = batch
        
        encoded_input = self.encoder(batched_sentences, batched_attn_masks)

        projected_input = self.input_projector(encoded_input)
        additional_projected_input = self.additional_input_projector(encoded_input)
        network_output = torch.concat((projected_input, additional_projected_input), dim = 1)

        encoded_types = self.type_encoder(batched_labels)
        
        return network_output, encoded_types
