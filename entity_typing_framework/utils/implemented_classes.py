from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork
from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import DistilBERTEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset


IMPLEMENTED_CLASSES = {
    'BaseEntityTypingNetwork': BaseEntityTypingNetwork,
    'DistilBERTEncoder' : DistilBERTEncoder,
    'OneHotTypeEncoder' : OneHotTypeEncoder,
    'Classifier' : Classifier,
    'BaseBERTTokenizedDataset' : BaseBERTTokenizedDataset
}