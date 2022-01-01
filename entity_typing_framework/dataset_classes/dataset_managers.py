

from entity_typing_framework.dataset_classes.datasets import BaseDataset
from entity_typing_framework.dataset_classes.datasets_for_dataloader import ET_Dataset
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset
from torch.utils.data.dataloader import DataLoader

class DatasetManager():
    def __init__(self, dataset_paths, tokenizer_params, dataloader_params) -> None:
        self.dataset_paths = dataset_paths
        self.tokenizer_params = tokenizer_params
        self.dataloader_params = dataloader_params

        self.dataset_routine()

    def dataset_routine(self):
        datasets = BaseDataset(dataset_paths = self.dataset_paths)
        
        self.create_type2id({partition_name : partition.labels for partition_name, partition in datasets.partitions.items()})
        
        tokenized_datasets = {partition_name: BaseBERTTokenizedDataset(dataset,
                                                                    self.type2id,
                                                                    **self.tokenizer_params['init_args']
                                                                    ) for partition_name, dataset in datasets.partitions.items()
                            }
        
        dataloader_ready_datasets = {partition_name : ET_Dataset(tokenized_dataset = t) for partition_name, t in tokenized_datasets.items()}

        self.dataloaders = {partition_name: DataLoader(dataset_obj,
                                                    **self.dataloader_params[partition_name]['init_args'])                        
                        for partition_name, dataset_obj in dataloader_ready_datasets.items()}
    
    def create_type2id(self, types_dict):
        partition_unique_types = {partition_name : set.union(*[set(t) for t in types]) for partition_name, types in types_dict.items()}
        dataset_unique_types = set.union(*list(partition_unique_types.values()))

        self.type2id = {t: id for id, t in enumerate(dataset_unique_types)}
        self.id2type = {id: t for t, id in self.type2id.items()}