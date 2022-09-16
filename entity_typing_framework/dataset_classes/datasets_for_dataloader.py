from torch.utils.data import Dataset
import numpy as np

class ET_Dataset(Dataset):
    '''
    This classes organizes the tokenized data inside a :doc:`dataset_tokenizer<dataset_tokenizers>` into tensors in order to be ready to fed a `Dataloader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
    
    This dataset saves tokenized_sentences in a single tensor, so it is adapt to be used with Bert-based tokenizers/encoders 

    Parameters:
        tokenized_dataset:
            an instance of a :doc:`Dataset Tokenizer <dataset_tokenizers>`
    
    Attributes:
        tokenized_sentences:
            numpy array of tokenized sentences extracted from :code:`tokenized_dataset`
        
        attn_masks:
            numpy array of attention masks extracted from :code:`tokenized_dataset`
        
        one_hot_types:
            numpy array of one hot encoding of types extracted from :code:`tokenized_dataset`
    '''

    def __init__(self, tokenized_dataset):
        super().__init__()

        self.input_ids = np.asarray(tokenized_dataset.tokenized_data['tokenized_sentences']['input_ids'])
        self.attn_masks = np.asarray(tokenized_dataset.tokenized_data['tokenized_sentences']['attention_mask'])
        self.one_hot_types = np.asarray(tokenized_dataset.tokenized_data['one_hot_types'])
    
    def __getitem__(self, idx):
        '''
        Override of :code:`pytorch.utils.Dataset.__getitem__()`, returns a batch when called

        Parameters:
            idx:
                integer index of the element to be returned, see the `official pytorch Dataset Documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
        
        Return:
            batch:
                a tuple composed of the :code:`idx`-esim element of :code:`tokenized_sencence`, the :code:`idx`-esim element of :code:`attn_masks`, and the :code:`idx`-esim element of :code:`one_hot_types` 

        '''
        return self.input_ids[idx], self.attn_masks[idx], self.one_hot_types[idx]

    def __len__(self):
        '''
        Override of :code:`pytorch.utils.Dataset.__len__()`, returns the number of elements in each attribute

        Return:
             the number of elements in each attribute (a single integer)
        '''
        return len(self.input_ids)

class ELMo_ET_Dataset(Dataset):
    '''
    This classes organizes the tokenized data inside a :doc:`dataset_tokenizer<dataset_tokenizers>` into tensors in order to be ready to fed a `Dataloader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
    
    This dataset saves tokenized_sentences in a single tensor, so it is adapt to be used with Bert-based tokenizers/encoders 

    Parameters:
        tokenized_dataset:
            an instance of a :doc:`Dataset Tokenizer <dataset_tokenizers>`
    
    Attributes:
        tokenized_sentences:
            numpy array of tokenized sentences extracted from :code:`tokenized_dataset`
        
        mention_mask:
            numpy array of mention masks extracted from :code:`tokenized_dataset`
        
        one_hot_types:
            numpy array of one hot encoding of types extracted from :code:`tokenized_dataset`
    '''

    def __init__(self, tokenized_dataset):
        super().__init__()

        self.input_ids = np.asarray(tokenized_dataset.tokenized_data['tokenized_sentences']['input_ids'])
        self.mention_mask = np.asarray(tokenized_dataset.tokenized_data['tokenized_sentences']['mention_mask'])
        self.one_hot_types = np.asarray(tokenized_dataset.tokenized_data['one_hot_types'])
    
    def __getitem__(self, idx):
        '''
        Override of :code:`pytorch.utils.Dataset.__getitem__()`, returns a batch when called

        Parameters:
            idx:
                integer index of the element to be returned, see the `official pytorch Dataset Documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
        
        Return:
            batch:
                a tuple composed of the :code:`idx`-esim element of :code:`tokenized_sencence`, the :code:`idx`-esim element of :code:`attn_masks`, and the :code:`idx`-esim element of :code:`one_hot_types` 

        '''
        return self.input_ids[idx], self.mention_mask[idx], self.one_hot_types[idx]

    def __len__(self):
        '''
        Override of :code:`pytorch.utils.Dataset.__len__()`, returns the number of elements in each attribute

        Return:
             the number of elements in each attribute (a single integer)
        '''
        return len(self.input_ids)
