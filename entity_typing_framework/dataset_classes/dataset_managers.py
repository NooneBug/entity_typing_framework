from pytorch_lightning.core.datamodule import LightningDataModule
from entity_typing_framework.utils.implemented_classes_lvl0 import IMPLEMENTED_CLASSES_LVL0

class DatasetManager(LightningDataModule):
    '''
    This module manages the dataset acquisition, the dataset tokenization and the dataloader creation.
    This module is able to use the following submodules:
    
    dataset_reader
        :ref:`entity_typing_framework.dataset_classes.datasets.BaseDataset <BaseDataset>`
    dataset_tokenizer
        :ref:`entity_typing_framework.dataset_classes.tokenized_datasets.BaseBERTTokenizedDataset <BaseBERTTokenizedDataset>`
    dataset
        :ref:`entity_typing_framework.dataset_classes.datasets_for_dataloader.ET_Dataset <ET_Dataset>`

    The :code:`__init__()` stores the parameters using :code:`self`, then calls :code:`read_datasets()` and then saves the return of :code:`get_type_number()` in type_number

    parameters
        dataset_paths
            A dictionary defined in the :code:`yaml` config file containing the dataset partititions names and paths, commonly in the format :code:`{name : path}`.

            See the classes in :doc:`Dataset Readers<dataset_readers>` for more information about format of dataset_paths
        
        tokenizer_params
            A dictionary defined in the :code:`yaml` config file used to instantiate the dataset_tokenizer submodule

            See the classes in :doc:`Dataset Tokenizers<dataset_tokenizers>` for more information about the expected params
        
        dataset_params
            A dictionary defined in the :code:`yaml` config file used to instantiate the dataset submodule

            See the classes in :doc:`Datasets<datasets>` for more information about the expected params

        dataloader_params
            A dictionary defined in the :code:`yaml` config file used to drive the instantiation of the :code:`torch.utils.data.dataloader.DataLoader`

            See the official torch documentation for `DataLoaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ for more information about the expected params
    '''
    def __init__(self, dataset_paths : dict, dataset_reader_params : dict, tokenizer_params : dict, dataset_params : dict, dataloader_params : dict):
        self.dataset_paths = dataset_paths
        self.dataset_reader_params = dataset_reader_params
        self.tokenizer_params = tokenizer_params
        self.datasets_params = dataset_params
        self.dataloader_params = dataloader_params
        self.read_datasets()
        self.type_number = self.get_type_number()

    def read_datasets(self):
        '''
        Instances a :doc:`dataset_reader <dataset_readers>` for each partition in :code:`dataset_path`, then calls :code:`create_type2id()`

        The class for dataset reader is chosen following the configuration file value under the keys :code:`data -> dataset_reader_params -> name`
        '''
        self.datasets = IMPLEMENTED_CLASSES_LVL0[self.dataset_reader_params['name']](dataset_paths = self.dataset_paths)
        
        self.create_type2id({partition_name : partition.labels for partition_name, partition in self.datasets.partitions.items()})
        
    def prepare_data(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.prepare_data <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#prepare-data>`_

        Tokenize each partition of the dataset with a :doc:`dataset_tokenizer <dataset_tokenizers>` class, chosen following the configuration file value under the keys :code:`data -> tokenizer_params -> name`
        '''
        self.tokenized_datasets = {partition_name: IMPLEMENTED_CLASSES_LVL0[self.tokenizer_params['name']](dataset,
                                                                            self.type2id,
                                                                            **self.tokenizer_params
                                                                            ) for partition_name, dataset in self.datasets.partitions.items()
                            }
    
    def setup(self, **kwargs):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.setup <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#prepare-data>`_

        Instance the :doc:`dataset <datasets>` and the `DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ for each partition.

        The :doc:`dataset <datasets>` class is chosen following the configuration file value under the keys :code:`data -> dataset_params -> name`

        The `DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ class is chosen following the configuration file value under the keys :code:`data -> dataloader_params -> name`

        '''
        dataloader_ready_datasets = {partition_name : IMPLEMENTED_CLASSES_LVL0[self.datasets_params['name']](tokenized_dataset = t) for partition_name, t in self.tokenized_datasets.items()}

        self.dataloaders = {partition_name: IMPLEMENTED_CLASSES_LVL0[self.datasets_params['name']](dataset_obj,
                                                    **self.dataloader_params[partition_name])                        
                        for partition_name, dataset_obj in dataloader_ready_datasets.items()}
    
    def create_type2id(self, types_dict):
        '''
        Creates and store the translation dictionaries :code:`type2id` and :code:`id2type`

        These dictionaries are used to translate from alphabetical version of a type to its token and vice versa

        These dictionaries are the only way to translate a type between alphabetical and token in all the framework
        '''
        partition_unique_types = {partition_name : set.union(*[set(t) for t in types]) for partition_name, types in types_dict.items()}
        dataset_unique_types = set.union(*list(partition_unique_types.values()))

        self.type2id = {t: id for id, t in enumerate(dataset_unique_types)}
        self.id2type = {id: t for t, id in self.type2id.items()}

    def get_type_number(self):
        '''
        Returns the number of the types in the whole dataset (all types in all partitions)
        '''
        return len(self.type2id)

    def train_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.train_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#train-dataloader>`_

        Returns the dataset partition used to train a model
        '''
        return self.dataloaders['train']
    
    def val_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#val-dataloader>`_

        Returns the dataset partition used to validate a model during training
        '''
        
        return self.dataloaders['dev']
    
    def test_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.test_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#test-dataloader>`_

        Returns the dataset partition used to test a model
        '''
        
        return self.dataloaders['test']
    
    