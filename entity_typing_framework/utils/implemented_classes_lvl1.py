from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNClassifier, KENNClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import BERTEncoder, DistilBERTEncoder, AdapterDistilBERTEncoder, AdapterBERTEncoder, BaseBERTLikeEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import BoxEmbeddingProjector, Classifier, ClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder

IMPLEMENTED_CLASSES_LVL1 = {
    'BERTEncoder' : BERTEncoder,
    'DistilBERTEncoder' : DistilBERTEncoder,
    'AdapterDistilBERTEncoder' : AdapterDistilBERTEncoder,
    'AdapterBERTEncoder' : AdapterBERTEncoder,
    'OneHotTypeEncoder' : OneHotTypeEncoder,
    'Classifier' : Classifier,
    'KENNClassifier' : KENNClassifier,
    'KENNClassifierForIncrementalTraining' : KENNClassifierForIncrementalTraining,
    'BoxEmbeddingProjector': BoxEmbeddingProjector,
    'ClassifierForIncrementalTraining' : ClassifierForIncrementalTraining
}