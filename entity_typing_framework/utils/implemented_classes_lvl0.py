from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork, IncrementalEntityTypingNetwork
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset, ELMoTokenizedDataset
from entity_typing_framework.main_module.NFETC_loss_modules.nfetc_loss_modules import BCENFETCCustomLossModule, BCENFETCLossModule
from entity_typing_framework.main_module.inference_manager import BaseInferenceManager, FlatToHierarchyThresholdOrMaxInferenceManager, IncrementalThresholdOrMaxInferenceManager, MaxInferenceManager, ThresholdOrMaxInferenceManager
from entity_typing_framework.main_module.losses_modules import BCELossModule, FlatBCELossModule, FlatRankingLossModule, RankingLossModule
from entity_typing_framework.main_module.KENN_loss_modules.kenn_loss_modules import BCEMultiLossModule
from torch.utils.data.dataloader import DataLoader
from entity_typing_framework.dataset_classes.datasets_for_dataloader import ELMo_ET_Dataset, ET_Dataset
from entity_typing_framework.dataset_classes.datasets import BaseDataset

IMPLEMENTED_CLASSES_LVL0 = {
    'BaseDataset' : BaseDataset,
    'BaseEntityTypingNetwork' : BaseEntityTypingNetwork,
    'IncrementalEntityTypingNetwork' : IncrementalEntityTypingNetwork,
    'BaseBERTTokenizedDataset' : BaseBERTTokenizedDataset,
    'torch.DataLoader' : DataLoader,
    'ET_Dataset' : ET_Dataset,
    'BaseInference' : BaseInferenceManager,
    'MaxInference' : MaxInferenceManager,
    'ThresholdOrMaxInference' : ThresholdOrMaxInferenceManager,
    'IncrementalThresholdOrMaxInference' : IncrementalThresholdOrMaxInferenceManager,
    'FlatToHierarchyThresholdOrMaxInference' : FlatToHierarchyThresholdOrMaxInferenceManager,
    'BCELossModule' : BCELossModule,
    'BCEMultiLossModule' : BCEMultiLossModule,
    'BCENFETCLossModule' : BCENFETCLossModule,
    'BCENFETCCustomLossModule' : BCENFETCCustomLossModule,
    'RankingLossModule' : RankingLossModule,
    'FlatBCELossModule' : FlatBCELossModule,
    'FlatRankingLossModule' : FlatRankingLossModule,
    'ELMoTokenizedDataset' : ELMoTokenizedDataset,
    'ELMo_ET_Dataset' : ELMo_ET_Dataset
    # 'BoxEmbeddingBCELoss' : BoxEmbeddingLogProbBCELoss,
    }
