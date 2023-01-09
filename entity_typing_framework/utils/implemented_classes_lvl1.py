from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNClassifier, KENNClassifierForIncrementalTraining, KENNClassifierForIncrementalTrainingOntonotes
# from entity_typing_framework.EntityTypingNetwork_classes.NFETC_modules.nfetc_classifier import NFETCClassifier
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_kenn_modules.box_embeddings_kenn import BoxEmbeddingKENNProjector
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxEmbeddingProjector, BoxEmbeddingIncrementalProjector
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector_fixed import BoxEmbeddingProjectorFixed, BoxEmbeddingProjectorFixedConstrained
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.vector_projector import VectorEmbeddingIncrementalProjector, VectorEmbeddingProjector
from entity_typing_framework.EntityTypingNetwork_classes.input_encoders import BERTEncoder, DistilBERTEncoder, AdapterDistilBERTEncoder, AdapterBERTEncoder, ELMoEncoder, GloVeEncoder, LSTMGloVeEncoder
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForIncrementalTraining
from entity_typing_framework.EntityTypingNetwork_classes.type2vec_modules.type2vec_projector import Type2VecProjector
from entity_typing_framework.EntityTypingNetwork_classes.type_encoders import OneHotTypeEncoder
from entity_typing_framework.main_module.box_losses.losses import BCEWithLogProbLoss
from torch.nn import BCELoss, CosineEmbeddingLoss

IMPLEMENTED_CLASSES_LVL1 = {
    'BERTEncoder' : BERTEncoder,
    'DistilBERTEncoder' : DistilBERTEncoder,
    'AdapterDistilBERTEncoder' : AdapterDistilBERTEncoder,
    'AdapterBERTEncoder' : AdapterBERTEncoder,
    'OneHotTypeEncoder' : OneHotTypeEncoder,
    'Classifier' : Classifier,
    'KENNClassifier' : KENNClassifier,
    'KENNClassifierForIncrementalTraining' : KENNClassifierForIncrementalTraining,
    'KENNClassifierForIncrementalTrainingOntonotes' : KENNClassifierForIncrementalTrainingOntonotes,
    'BoxEmbeddingProjector': BoxEmbeddingProjector,
    'BoxEmbeddingProjectorFixed': BoxEmbeddingProjectorFixed,
    'BoxEmbeddingProjectorFixedConstrained': BoxEmbeddingProjectorFixedConstrained,
    'VectorEmbeddingProjector': VectorEmbeddingProjector,
    'ClassifierForIncrementalTraining' : ClassifierForIncrementalTraining,
    'BoxEmbeddingIncrementalProjector' : BoxEmbeddingIncrementalProjector,
    'VectorEmbeddingIncrementalProjector' : VectorEmbeddingIncrementalProjector,
    'BoxEmbeddingKENNProjector' : BoxEmbeddingKENNProjector,
    'Type2VecProjector' : Type2VecProjector,
    # 'NFETCClassifier' : NFETCClassifier,
    'BCELoss' : BCELoss,
    'BCEWithLogProbLoss' : BCEWithLogProbLoss,
    'CosineEmbeddingLoss' : CosineEmbeddingLoss,
    'ELMoEncoder' : ELMoEncoder,
    'GloVeEncoder' : GloVeEncoder,
    'LSTMGloVeEncoder' : LSTMGloVeEncoder
}