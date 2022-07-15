from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork, IncrementalEntityTypingNetwork
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset
from entity_typing_framework.main_module.inference_manager import BaseInferenceManager, IncrementalThresholdOrMaxInferenceManager, ThresholdOrMaxInferenceManager
from entity_typing_framework.main_module.losses_modules import BCELossModule
from entity_typing_framework.main_module.KENN_loss_modules.kenn_loss_modules import BCEMultiLossModule
from torch.utils.data.dataloader import DataLoader
from entity_typing_framework.dataset_classes.datasets_for_dataloader import ET_Dataset
from entity_typing_framework.dataset_classes.datasets import BaseDataset

IMPLEMENTED_CLASSES_LVL0 = {
    'BaseDataset' : BaseDataset,
    'BaseEntityTypingNetwork' : BaseEntityTypingNetwork,
    'IncrementalEntityTypingNetwork' : IncrementalEntityTypingNetwork,
    'BaseBERTTokenizedDataset' : BaseBERTTokenizedDataset,
    'torch.DataLoader' : DataLoader,
    'ET_Dataset' : ET_Dataset,
    'BaseInference' : BaseInferenceManager,
    'ThresholdOrMaxInference' : ThresholdOrMaxInferenceManager,
    'IncrementalThresholdOrMaxInference' : IncrementalThresholdOrMaxInferenceManager,
    'BCELossModule' : BCELossModule,
    'BCEMultiLossModule' : BCEMultiLossModule,
    # 'BoxEmbeddingBCELoss' : BoxEmbeddingLogProbBCELoss,
    }
