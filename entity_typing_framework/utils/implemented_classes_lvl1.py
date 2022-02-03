from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import DistilBERTEncoder, AdapterDistilBERTEncoder, BaseBERTLikeEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder


IMPLEMENTED_CLASSES_LVL1 = {
    'DistilBERTEncoder' : DistilBERTEncoder,
    'AdapterDistilBERTEncoder' : AdapterDistilBERTEncoder,
    'OneHotTypeEncoder' : OneHotTypeEncoder,
    'Classifier' : Classifier,
}