

# from entity_typing_framework.dataset_classes.dataloaders import BaseDataLoader
from entity_typing_framework.dataset_classes.datasets import BaseDataset
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset


class DatasetManager():
    def __init__(self, dataset_paths, tokenizer_params) -> None:
        self.dataset_paths = dataset_paths
        self.tokenizer_params = tokenizer_params

        self.dataset_routine()

    def dataset_routine(self):
        datasets = BaseDataset(dataset_paths = self.dataset_paths)
        
        tokenized_datasets = {dataset_name: BaseBERTTokenizedDataset(dataset, 
                                                                    **self.tokenizer_params['init_args']
                                                                    ) for dataset_name, dataset in datasets.partitions.items()
                            } 

        # dataloaders = {partition_name: BaseDataLoader(dataset_obj) for partition_name, dataset_obj in datasets.items()}