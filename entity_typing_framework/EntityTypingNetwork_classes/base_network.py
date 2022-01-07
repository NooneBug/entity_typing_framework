from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import DistilBERTEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder
from pytorch_lightning.core.lightning import LightningModule
from entity_typing_framework.utils.implemented_classes import IMPLEMENTED_CLASSES

class BaseEntityTypingNetwork(LightningModule):
    def __init__(self, network_params, type_number
    # , encoder_params, type_encoder_params, 
    # inference_params, metric_manager_params, loss_params, 
    ) -> None:
        super().__init__()

        encoder_params = network_params['encoder_params']
        self.encoder = IMPLEMENTED_CLASSES[encoder_params['name']](**encoder_params)

        type_encoder_params = network_params['type_encoder_params']
        self.type_encoder = IMPLEMENTED_CLASSES[type_encoder_params['name']](type_number=type_number, **type_encoder_params)

        input_projector_params = network_params['input_projector_params']
        self.input_projector = IMPLEMENTED_CLASSES[input_projector_params['name']](type_number=type_number, 
                                            input_dim = self.encoder.get_representation_dim(), 
                                            layers_parameters = input_projector_params)

    def forward(self, batch):
        batched_sentences, batched_attn_masks, batched_labels = batch
        
        encoded_input = self.encoder(batched_sentences, batched_attn_masks)
        projected_input = self.input_projector(encoded_input)
        encoded_types = self.type_encoder(batched_labels)
        
        return projected_input, encoded_types

