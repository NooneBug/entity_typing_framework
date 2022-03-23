from entity_typing_framework.EntityTypingNetwork_classes.base_network import BaseEntityTypingNetwork
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset
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
    'ET_Dataset' : ET_Dataset
}