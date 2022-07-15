from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNClassifier, KENNClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_kenn_modules.box_embeddings_kenn import BoxEmbeddingKENNProjector
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxEmbeddingProjector, BoxEmbeddingIncrementalProjector
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector_fixed import BoxEmbeddingProjectorFixed, BoxEmbeddingProjectorFixedConstrained
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.vector_projector import VectorEmbeddingIncrementalProjector, VectorEmbeddingProjector
from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import BERTEncoder, DistilBERTEncoder, AdapterDistilBERTEncoder, AdapterBERTEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder
from entity_typing_framework.main_module.box_losses.losses import BCEWithLogProbLoss
from torch.nn import BCELoss

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
    'BoxEmbeddingProjectorFixed': BoxEmbeddingProjectorFixed,
    'BoxEmbeddingProjectorFixedConstrained': BoxEmbeddingProjectorFixedConstrained,
    'VectorEmbeddingProjector': VectorEmbeddingProjector,
    'ClassifierForIncrementalTraining' : ClassifierForIncrementalTraining,
    'BoxEmbeddingIncrementalProjector' : BoxEmbeddingIncrementalProjector,
    'VectorEmbeddingIncrementalProjector' : VectorEmbeddingIncrementalProjector,
    'BoxEmbeddingKENNProjector' : BoxEmbeddingKENNProjector,
    'BCELoss' : BCELoss,
    'BCEWithLogProbLoss' : BCEWithLogProbLoss
}