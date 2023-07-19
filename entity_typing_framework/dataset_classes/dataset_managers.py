from pytorch_lightning.core.datamodule import LightningDataModule
from entity_typing_framework.utils.implemented_classes_lvl0 import IMPLEMENTED_CLASSES_LVL0
from pytorch_lightning.trainer.supporters import CombinedLoader
from transformers import AutoTokenizer
import pickle
import os

RW_OPTIONS_MODALITY = ['Create', 'CreateAndSave', 'Load']

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

            The dataset paths are expected to be in the :code:`yaml` config file under the key : :code:`data.dataset_paths`

            See the classes in :doc:`Dataset Readers<dataset_readers>` for more information about format of dataset_paths
        
        tokenizer_params
            A dictionary defined in the :code:`yaml` config file used to instantiate the dataset_tokenizer submodule

            The tokenizer parameters are expected to be in the :code:`yaml` config file under the key : :code:`data.tokenizer_params`

            See the classes in :doc:`Dataset Tokenizers<dataset_tokenizers>` for more information about the expected params
        
        dataset_params
            A dictionary defined in the :code:`yaml` config file used to instantiate the dataset submodule

            The dataset parameters are expected to be in the :code:`yaml` config file under the key : :code:`data.dataset_params`

            See the classes in :doc:`Datasets<datasets>` for more information about the expected params

        dataloader_params
            A dictionary defined in the :code:`yaml` config file used to drive the instantiation of the :code:`torch.utils.data.dataloader.DataLoader`

            The dataloader parameters are expected to be in the :code:`yaml` config file under the key : :code:`data.dataloader_params`

            See the official torch documentation for `DataLoaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ for more information about the expected params
        
        rw_options
            A dictionary defined in the :code:`yaml` config file used to set the read/write options to load and save the tokenized datasets

            The read/write options are expected to be in the :code:`yaml` config file under the key : :code:`data.rw_options`
            
            rw_options is a dictionary composed of the following keys:
                :code:`modality` specifies the options for load and/or save the object: 
                    
                    #. :code:`Create` to create the tokenized version of the dataset, 
                    #. :code:`CreateAndSave` to save it, 
                    #. :code:`Load` to load it,

                :code:`dirpath` defines the read/write directory from which read or write, 
                
                :code:`light` enables the save/load of the tokenized dataset w/o the original data and tokenizer (see :code:`make_light` method in :ref:`entity_typing_framework.dataset_classes.tokenized_datasets.BaseBERTTokenizedDataset <BaseBERTTokenizedDataset>`)


    '''
    def __init__(self, dataset_paths : dict, dataset_reader_params : dict, tokenizer_params : dict, dataset_params : dict, dataloader_params : dict,
                rw_options : dict, truncation_side = ''):
        self.dataset_paths = dataset_paths
        self.dataset_reader_params = dataset_reader_params
        self.tokenizer_params = tokenizer_params
        self.datasets_params = dataset_params
        self.dataloader_params = dataloader_params
        self.rw_options = rw_options
        self.rw_options['modality'] = self.rw_options['modality'].lower()
        if self.rw_options['modality'] == 'load':
            self.create_type2id_from_file()
        elif self.rw_options['modality'] == 'create' or self.rw_options['modality'] == 'createandsave':
            self.read_datasets()
        else:
            raise Exception(f"Error: the value of data.rw_options.modality must be in {RW_OPTIONS_MODALITY}")
        self.type_number = self.get_type_number()
        self.tokenizer_config_name = self.get_tokenizer_config_name()
        self.truncation_side = truncation_side
        tokenizer = self.instance_tokenizer(**self.tokenizer_params)
        self.mask_token_id = tokenizer.mask_token_id

    def read_datasets(self):
        '''
        Instances a :doc:`dataset_reader <dataset_readers>` for each partition in :code:`dataset_path`, then calls :code:`create_type2id()`

        The class for dataset reader is chosen following the configuration file value under the key :code:`data.dataset_reader_params.name`
        '''
        
        if "types_list_path" in self.rw_options:
            # allows the usage of an external vocabulary of types
            self.create_type2id_from_file()
            self.datasets = IMPLEMENTED_CLASSES_LVL0[self.dataset_reader_params['name']](dataset_paths = self.dataset_paths, preexistent_type2id = self.type2id, **self.dataset_reader_params)      
        else:
            # defines the vocabulary of types based on all the partitions defined in the yaml
            self.datasets = IMPLEMENTED_CLASSES_LVL0[self.dataset_reader_params['name']](dataset_paths = self.dataset_paths, **self.dataset_reader_params)
            self.create_type2id_from_partitions({partition_name : partition.labels for partition_name, partition in self.datasets.partitions.items()})
        
        self.save_type2id()

    def prepare_data(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.prepare_data <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#prepare-data>`_

        If :code:`data.rw_options.modality is 'Create'` tokenize each partition of the dataset with a :doc:`dataset_tokenizer <dataset_tokenizers>` class, chosen following the configuration file value under the key :code:`data.tokenizer_params.name`
        
        If :code:`data.rw_options.modality is 'CreateAndSave'` has the same behavior of :code:`Create` but also saves each instance of :doc:`dataset_tokenizer <dataset_tokenizers>` following :code:`data.rw_options.dirpath` and :code:`data.rw_option.light`  
        
        If :code:`data.rw_options.modality is 'Load'` loads a previously saved instance of each :doc:`dataset_tokenizer <dataset_tokenizers>` following :code:`data.rw_options.dirpath` and :code:`data.rw_option.light`  
        '''

        if self.rw_options['modality'] == 'load': 
            self.load_tokenized_datasets()
        else: 
            # create
            # tokenizer = self.instance_tokenizer(bertlike_model_name = self.tokenizer_params['bertlike_model_name'])
            tokenizer = self.instance_tokenizer(**self.tokenizer_params)
            self.tokenized_datasets = {partition_name: IMPLEMENTED_CLASSES_LVL0[self.tokenizer_params['name']](
                                                                                dataset=dataset,
                                                                                type2id=self.type2id,
                                                                                tokenizer=tokenizer,
                                                                                partition_name=partition_name,
                                                                                **self.tokenizer_params
                                                                                ) for partition_name, dataset in self.datasets.partitions.items()
                                }
            # save
            if self.rw_options['modality'] == 'createandsave':
                self.save_tokenized_datasets()
            # free original datasets
            if self.rw_options['light']:
                self.datasets = None
    
    def save_tokenized_datasets(self):
        '''
        saves a pickle object that contains the :code:`tokenized_dataset`, the :code:`tokenizer_params`, and the :code:`rw_options.light` parameters

        the object is saved folowing the path composed by :code:`rw_options.dirpath` and the method :code:`get_tokenizer_config_name`
        '''
        # delete tokenizer and dataset from tokenized_datasets
        if self.rw_options['light']:
            for _, tokenized_dataset in self.tokenized_datasets.items():
                tokenized_dataset.make_light()
            
        # prepare object to save
        data = {
            'tokenized_datasets' : self.tokenized_datasets,
            'tokenizer_params' : self.tokenizer_params,
            'light' : self.rw_options['light'],
        }
        # save
        path_to_save = os.path.join(self.rw_options['dirpath'], f'{self.tokenizer_config_name}.pickle')
        if not os.path.exists(self.rw_options['dirpath']):
            os.makedirs(self.rw_options['dirpath'])
        with open(path_to_save, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenized_datasets(self):
        '''
        loads an object saved with :code:`save_tokenized_datasets`, the object is loaded following the path obtained by :code:`rw_options.dirpath` and the method :code:`get_tokenizer_config_name`
        '''
        # load
        path_to_load = os.path.join(self.rw_options['dirpath'], f'{self.tokenizer_config_name}.pickle')
        with open(path_to_load, 'rb') as f:
            data = pickle.load(f)
        # check consistency between the format of the data to load and the 'light' option
        # NOTE: this should not happens since the filename has the 'light' info
        if self.rw_options['light'] != data['light']:
            raise Exception(f"Error: you are trying to load {path_to_load} with 'light' set to {self.rw_options['light']},\
                            but the tokenized datasets are saved with 'light' set to {data['light']}")
        # load datasets and mappings
        self.tokenized_datasets = data['tokenized_datasets']

    def save_type2id(self):
        '''
        saves a pickle object that contains :code:`self.type2id` and :code:`self.id2type`, the file into save is obtained by :code:`rw_options.dirpath\\type2id.pickle` 
        '''
        
        if not os.path.exists(self.rw_options['dirpath']):
            os.makedirs(self.rw_options['dirpath'])
        
        path_to_save = os.path.join(self.rw_options['dirpath'], 'types_list.txt')        
        with open(path_to_save, 'w') as f:
            for k in self.type2id.keys():
                f.write(k + '\n')

        print(f'{len(self.type2id)} types have been saved in {path_to_save}. These types will be used during this run')

    def get_tokenizer_config_name(self):
        '''
        generates the name for the output file using :code:`self.tokenizer_params` and :code:`self.rw_option`
        '''
        config_name = self.tokenizer_params['bertlike_model_name']
        config_name += '_'
        config_name += f"M{self.tokenizer_params['max_mention_words']}"
        config_name += f"L{self.tokenizer_params['max_left_words']}"
        config_name += f"R{self.tokenizer_params['max_right_words']}"
        config_name += f"T{self.tokenizer_params['max_tokens']}"
        config_name += '_light' if self.rw_options['light'] else ''
        config_name = config_name.replace('/', '__')
        return config_name

    def instance_tokenizer(self, bertlike_model_name, **kwargs):
        '''
        instance a tokenizer with `huggingface AutoTokenizer <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer>`_

        parameters:
            bertlike_model_name:
                see :code:`bertlike_model_name` in the documentation of the entire class
        
        return:
            an instance of `huggingface AutoTokenizer <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer>`_

        '''
        if self.truncation_side:
            return AutoTokenizer.from_pretrained(bertlike_model_name, truncation_side = self.truncation_side)
        else:
            return AutoTokenizer.from_pretrained(bertlike_model_name)


    def setup(self, **kwargs):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.setup <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#prepare-data>`_

        Instance the :doc:`dataset <datasets>` and the `DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ for each partition.

        The :doc:`dataset <datasets>` class is chosen following the configuration file value under the key :code:`data.dataset_params.name`

        The `DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ class is chosen following the configuration file value under the key :code:`data.dataloader_params.name`

        '''
        dataloader_ready_datasets = {partition_name : IMPLEMENTED_CLASSES_LVL0[self.datasets_params['name']](tokenized_dataset = t) for partition_name, t in self.tokenized_datasets.items()}

        self.dataloaders = {partition_name: IMPLEMENTED_CLASSES_LVL0[self.dataloader_params['name']](dataset_obj,
                                                    **self.dataloader_params[partition_name])                        
                        for partition_name, dataset_obj in dataloader_ready_datasets.items()}
    
    def create_type2id_from_file(self):
        if "types_list_path" in self.rw_options:
            file_path = self.rw_options['types_list_path']
        else:
            file_path = os.path.join(self.rw_options['dirpath'], 'types_list.txt')     
            
        with open(file_path, 'r') as inp:
            types = [l.replace('\n', '') for l in inp.readlines()]
        
        self.type2id = {k: i for i, k in enumerate(types)}
        self.id2type = {id: t for t, id in self.type2id.items()}

    def create_type2id_from_partitions(self, types_dict):
        '''
        Creates and store the translation dictionaries :code:`type2id` and :code:`id2type`

        These dictionaries are used to translate from alphabetical version of a type to its token and vice versa

        These dictionaries are the only way to translate a type between alphabetical and token in all the framework
        '''
        partition_unique_types = {partition_name : set.union(*[set(t) for t in types]) for partition_name, types in types_dict.items()}
        dataset_unique_types = sorted(set.union(*list(partition_unique_types.values())))

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
    

class IncrementalDatasetManager(DatasetManager):
    def create_type2id_from_partitions(self, types_dict):
        '''
        Creates and store the translation dictionaries :code:`type2id` and :code:`id2type`

        These dictionaries are used to translate from alphabetical version of a type to its token and vice versa

        These dictionaries are the only way to translate a type between alphabetical and token in all the framework
        '''
        partition_unique_types = {partition_name : set.union(*[set(t) for t in types]) for partition_name, types in types_dict.items()}
        
        dataset_unique_types = {}
        for name in ['pretraining', 'incremental']:
            dataset_unique_types[name] = set()
            for k, v in partition_unique_types.items():
                if name in k:
                    dataset_unique_types[name].update(v)
        dataset_unique_types = {k: sorted(v) for k, v in dataset_unique_types.items()}

        self.type2id = {t: id for id, t in enumerate(dataset_unique_types['pretraining'])}
        for k in dataset_unique_types['incremental']:
            if k not in self.type2id:
                self.type2id[k] = len(self.type2id)
    
        self.id2type = {id: t for t, id in self.type2id.items()}

    def train_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.train_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#train-dataloader>`_

        Returns the dataset partition used to train a model
        '''
        loaders = {'pretraining': self.dataloaders['pretraining_train']}
        loaders_incremental = {k: v for k,v in self.dataloaders.items() if 'incremental_train' in k}
        loaders.update(loaders_incremental)
        return CombinedLoader(loaders, mode='min_size')
    
    def val_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#val-dataloader>`_

        Returns the dataset partition used to validate a model during training
        '''
        loaders = [self.dataloaders['pretraining_dev']] + [v for k,v in self.dataloaders.items() if 'incremental_dev' in k]
        return loaders   

    def predict_dataloader(self):
        return self.test_dataloader

class IncrementalDatasetManagerWithTestLog(IncrementalDatasetManager):
    
    def __init__(self, dataset_paths: dict, dataset_reader_params: dict, tokenizer_params: dict, dataset_params: dict, dataloader_params: dict, rw_options: dict):
        super().__init__(dataset_paths, dataset_reader_params, tokenizer_params, dataset_params, dataloader_params, rw_options)
        self.test_index = len([d for d in self.dataset_paths if 'pretraining_dev' in d or 'incremental_dev' in d])

    def val_dataloader(self):
        '''
        '''
        loaders = [self.dataloaders['pretraining_dev']] + [v for k,v in self.dataloaders.items() if 'incremental_dev' in k]
        # loaders += [self.dataloaders['test']]
        return loaders
    
    def test_dataloader(self):
        '''
        Override of `pytorch_lightning.core.datamodule.LightningDataModule.val_dataloader <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#val-dataloader>`_

        Returns the dataset partition used to validate a model during training
        '''
        loaders = self.dataloaders['test']
        return loaders

from allennlp.modules.elmo import batch_to_ids

class ELMoDatasetManager(DatasetManager):

    def instance_tokenizer(self, **kwargs):
        return batch_to_ids

    def get_tokenizer_config_name(self):
        config_name = f"elmo_T{self.tokenizer_params['max_tokens']}"
        config_name += '_light' if self.rw_options['light'] else ''

        return config_name

from tqdm import tqdm
import numpy as np

class GloVeDatasetManager(DatasetManager):

    def __init__(self, dataset_paths: dict, dataset_reader_params: dict, tokenizer_params: dict, dataset_params: dict, dataloader_params: dict, rw_options: dict):
        
        super().__init__(dataset_paths, dataset_reader_params, tokenizer_params, dataset_params, dataloader_params, rw_options)
        
        if self.rw_options['modality'] == 'load':

            # we dont need glove vectors since we directly load a tokenized dataset

            # folder_path = self.rw_options['dirpath']
            # with open(os.path.join(folder_path, self.get_tokenizer_config_name() + '_glove_vectors.pkl'), 'rb') as inp:
            #     glove_vectors = pickle.load(inp)

            self.embs_npa = {}
            self.vocab = {}

        elif self.rw_options['modality'] == 'create' or self.rw_options['modality'] == 'createandsave':
        
            path = tokenizer_params['glove_path']

            vocab,embeddings = [],[]

            with open(path,'rt') as fi:
                full_content = fi.read().strip().split('\n')

            for i in tqdm(range(len(full_content)), desc=f'Reading GloVe vectors from {path}...'):
                i_word = full_content[i].split(' ')[0]
                i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
                vocab.append(i_word)
                embeddings.append(i_embeddings)

            embs_npa = np.array(embeddings)

            #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
            vocab = ['<pad>', '<unk>'] + vocab

            pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
            unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

            #insert embeddings for pad and unk tokens at top of embs_npa.
            self.embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
            self.vocab = {k: i for i, k in enumerate(vocab)}

            del embeddings, vocab
            
            if self.rw_options['modality'] == 'createandsave':
                folder_path = self.rw_options['dirpath']
                with open(os.path.join(folder_path, self.get_tokenizer_config_name() + '_glove_vectors.pkl'), 'wb') as out:
                    pickle.dump({'embs_npa' : self.embs_npa, 'vocab' : self.vocab}, out, protocol=pickle.HIGHEST_PROTOCOL)
                    

    def instance_tokenizer(self, **kwargs):
        return GloVeTokenizer(vectors=self.embs_npa, vocab = self.vocab)

    def get_tokenizer_config_name(self):
        config_name = f"glove_T{self.tokenizer_params['max_tokens']}"
        config_name += '_light' if self.rw_options['light'] else ''

        return config_name

import torch

class GloVeTokenizer():
    def __init__(self, vectors, vocab) -> None:
        self.global_vectors = vectors
        self.vocab = vocab
    
    def tokenize(self, input_sentence, max_len):
        tokenized = torch.full((max_len, 300), 0.)
        for i, word in enumerate(input_sentence):
            if i < max_len:
                if word in self.vocab:
                    tok = self.global_vectors[self.vocab[word]]
                else:
                    tok = self.global_vectors[self.vocab['<unk>']]
                tokenized[i] = tokenized[i] + tok # since tokenized is initialized with 0s we can sum
        return tokenized

    def tokenize_single_sentence(self, input_sentence):
        return input_sentence
    
class ALIGNIEDatasetManager(DatasetManager):
    def __init__(self, dataset_paths: dict, dataset_reader_params: dict, tokenizer_params: dict, dataset_params: dict, dataloader_params: dict, rw_options: dict, truncation_side='', verbalizer_path = ''):
        super().__init__(dataset_paths, dataset_reader_params, tokenizer_params, dataset_params, dataloader_params, rw_options, truncation_side)
        self.verbalizer_path = verbalizer_path
        self.create_verbalizer()
    
    def create_verbalizer(self):
        with open(self.verbalizer_path, 'r') as inp:
            lines = [l.replace('\n', '').split(':') for l in inp.readlines()]

        verbalizer = {l[0]: l[1].split(' ') for l in lines}
        self.tokenizer = self.instance_tokenizer(**self.tokenizer_params)
        
        self.verbalizer = {}

        for t in self.type2id:
            if t in verbalizer:
                values = verbalizer[t]
                ids = [t[0] for t in self.tokenizer([' ' + v for v in values], add_special_tokens=False)['input_ids']]
                self.verbalizer[self.type2id[t]] = ids
                
            else:
                raise Exception(f'Type {t} not in verbalizer')

        

