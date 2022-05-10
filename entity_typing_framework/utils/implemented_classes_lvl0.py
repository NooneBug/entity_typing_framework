from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork, EntityTypingNetworkForIncrementalTraining
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset
from entity_typing_framework.main_module.KENN_losses.kenn_losses import KENNBCEMultiloss
from entity_typing_framework.main_module.losses import BCELossForET
from torch.utils.data.dataloader import DataLoader
from entity_typing_framework.dataset_classes.datasets_for_dataloader import ET_Dataset
from entity_typing_framework.dataset_classes.datasets import BaseDataset
from entity_typing_framework.dataset_classes.KENN_datasets.kenn_dataset import KENNDataset

IMPLEMENTED_CLASSES_LVL0 = {
    'BaseDataset' : BaseDataset,
    'DatasetWithKENN' : KENNDataset,
    'BaseEntityTypingNetwork': BaseEntityTypingNetwork,
    'BaseBERTTokenizedDataset' : BaseBERTTokenizedDataset,
    'torch.DataLoader' : DataLoader,
    'ET_Dataset' : ET_Dataset,
    'BCELossForET' : BCELossForET,
    'KENNBCEMultiloss' : KENNBCEMultiloss,
    'EntityTypingNetworkForIncrementalTraining': EntityTypingNetworkForIncrementalTraining
    }