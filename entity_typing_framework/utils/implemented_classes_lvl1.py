from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNClassifier, KENNClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxEmbeddingProjector, BoxEmbeddingIncrementalProjector
from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import BERTEncoder, DistilBERTEncoder, AdapterDistilBERTEncoder, AdapterBERTEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForIncrementalTraining
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
    'ClassifierForIncrementalTraining' : ClassifierForIncrementalTraining,
    'BoxEmbeddingIncrementalProjector' : BoxEmbeddingIncrementalProjector,
}