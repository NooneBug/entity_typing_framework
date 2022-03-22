from pytorch_lightning.core.lightning import LightningModule
from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1

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
    def __init__(self, name, network_params, type_number
    # , encoder_params, type_encoder_params, 
    # inference_params, metric_manager_params, loss_params, 
    ):
        super().__init__()

        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES_LVL1[encoder_params['name']](**encoder_params)

        type_encoder_params = network_params['type_encoder_params']
        self.type_encoder = IMPLEMENTED_CLASSES_LVL1[type_encoder_params['name']](type_number=type_number, **type_encoder_params)

        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES_LVL1[input_projector_params['name']](type_number=type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
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