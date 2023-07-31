from entity_typing_framework.EntityTypingNetwork_classes.base_network import ALIGNIENetwork, BaseEntityTypingNetwork, CrossDatasetEntityTypingNetwork, IncrementalEntityTypingNetwork, PROMETNetwork
from entity_typing_framework.EntityTypingNetwork_classes.projectors import ALIGNIEProjector
from entity_typing_framework.dataset_classes.large_datasets.large_datasets_for_dataloaders import BERT_ET_DatasetLarge, ELMo_ET_DatasetLarge, ET_DatasetLarge
from entity_typing_framework.dataset_classes.large_datasets.large_tokenized_datasets import BaseBERTTokenizedDatasetLarge, ELMoTokenizedDatasetLarge
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset, ELMoTokenizedDataset, GloVeTokenizedDataset, MentionSentenceBERTTokenizedDataset, ALIGNIEPromptTokenizedDataset
from entity_typing_framework.main_module.NFETC_loss_modules.nfetc_loss_modules import BCENFETCCustomLossModule, BCENFETCLossModule
from entity_typing_framework.main_module.inference_manager import BaseInferenceManager, FlatToHierarchyThresholdOrMaxInferenceManager, IncrementalDoubleThresholdOrMaxInferenceManager, IncrementalThresholdOrMaxInferenceManager, MaxInferenceManager, MaxLeafInferenceManager, ThresholdOrMaxInferenceManager, ALIGNIEInferenceManager
from entity_typing_framework.main_module.losses_modules import BCELossModule, FlatBCELossModule, FlatRankingLossModule, RankingLossModule, WeightedBCELossModule, CELossModule, ALIGNIELossModule, KLDivLossModule
from entity_typing_framework.main_module.KENN_loss_modules.kenn_loss_modules import BCEMultiLossModule
from entity_typing_framework.main_module.metric_manager import MetricManager, LeavesMetricManager, ALIGNIEMetricManager
from torch.utils.data.dataloader import DataLoader
from entity_typing_framework.dataset_classes.datasets_for_dataloader import ELMo_ET_Dataset, ET_Dataset
from entity_typing_framework.dataset_classes.datasets import BaseDataset

IMPLEMENTED_CLASSES_LVL0 = {
    'BaseDataset' : BaseDataset,
    'BaseEntityTypingNetwork' : BaseEntityTypingNetwork,
    'CrossDatasetEntityTypingNetwork' : CrossDatasetEntityTypingNetwork,
    'IncrementalEntityTypingNetwork' : IncrementalEntityTypingNetwork,
    'BaseBERTTokenizedDataset' : BaseBERTTokenizedDataset,
    'torch.DataLoader' : DataLoader,
    'ET_Dataset' : ET_Dataset,
    'BaseInference' : BaseInferenceManager,
    'MaxInference' : MaxInferenceManager,
    'ThresholdOrMaxInference' : ThresholdOrMaxInferenceManager,
    'IncrementalThresholdOrMaxInference' : IncrementalThresholdOrMaxInferenceManager,
    'IncrementalDoubleThresholdOrMaxInference' : IncrementalDoubleThresholdOrMaxInferenceManager,
    'FlatToHierarchyThresholdOrMaxInference' : FlatToHierarchyThresholdOrMaxInferenceManager,
    'BCELossModule' : BCELossModule,
    'BCEMultiLossModule' : BCEMultiLossModule,
    'BCENFETCLossModule' : BCENFETCLossModule,
    'BCENFETCCustomLossModule' : BCENFETCCustomLossModule,
    'RankingLossModule' : RankingLossModule,
    'FlatBCELossModule' : FlatBCELossModule,
    'FlatRankingLossModule' : FlatRankingLossModule,
    'WeightedBCELossModule' : WeightedBCELossModule,
    'ELMoTokenizedDataset' : ELMoTokenizedDataset,
    'ELMo_ET_Dataset' : ELMo_ET_Dataset,
    'ET_DatasetLarge' : ET_DatasetLarge,
    'ELMo_ET_DatasetLarge' : ELMo_ET_DatasetLarge,
    'BERT_ET_DatasetLarge' : BERT_ET_DatasetLarge,
    'BaseBERTTokenizedDatasetLarge' : BaseBERTTokenizedDatasetLarge,
    'ELMoTokenizedDatasetLarge' : ELMoTokenizedDatasetLarge,
    'GloVeTokenizedDataset' : GloVeTokenizedDataset,
    'MaxLeafInference' : MaxLeafInferenceManager,
    'MetricManager' : MetricManager,
    'LeavesMetricManager' : LeavesMetricManager,
    'MentionSentenceBERTTokenizedDataset' : MentionSentenceBERTTokenizedDataset,
    'CELossModule' : CELossModule,
    # 'BoxEmbeddingBCELoss' : BoxEmbeddingLogProbBCELoss,
    'ALIGNIEPromptTokenizedDataset' : ALIGNIEPromptTokenizedDataset,
    'ALIGNIENetwork' : ALIGNIENetwork,
    'ALIGNIELossModule' : ALIGNIELossModule,
    'KLDivLossModule' : KLDivLossModule,
    'ALIGNIEInferenceManager': ALIGNIEInferenceManager,
    'PROMETNetwork': PROMETNetwork,
    'ALIGNIEMetricManager' : ALIGNIEMetricManager
    }
