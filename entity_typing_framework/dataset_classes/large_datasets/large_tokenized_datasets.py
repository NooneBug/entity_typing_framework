from entity_typing_framework.dataset_classes.datasets import BaseDataset
from entity_typing_framework.dataset_classes.tokenized_datasets import BaseBERTTokenizedDataset, ELMoTokenizedDataset
import shutil
import os
import math
import torch
from tqdm import tqdm

class SliceManager():
  
  # TODO: pass partition_name to GeneralTokenizedDataset
  def __init__(self, partition_name, slice_dirpath, slice_dimension, n_examples):
    self.slice_dimension = slice_dimension
    self.slice_dirpath = slice_dirpath
    self.partition_name = partition_name
    self.n_examples = n_examples
    

  def tokenize(self, original_sentences, processed_sentences, types, tokenize_fn, save_slice_fn):
    
    # prepare empty directory for the slices
    slice_partition_dirpath = os.path.join(self.slice_dirpath, self.partition_name)
    if os.path.exists(slice_partition_dirpath):
      shutil.rmtree(slice_partition_dirpath)
    os.makedirs(slice_partition_dirpath)
    
    # compute number of slices
    n_slices = math.ceil(self.n_examples / self.slice_dimension)

    # prepare mapping dict
    slice_mapping = {}

    # for each slice, tokenize sentences, save on disk and save slice-path mappings
    for i in tqdm(range(n_slices), desc='Tokenizing and saving slices'):
      # extract sentences and labels for the slice
      start_idx = i * self.slice_dimension
      end_idx = start_idx + self.slice_dimension

      slice_tokenized_data = tokenize_fn(original_sentences = original_sentences[start_idx:end_idx], 
                                          processed_sentences = processed_sentences[start_idx:end_idx], 
                                          types = types[start_idx:end_idx])

      # save slice
      slice_mapping.update(save_slice_fn(slice_tokenized_data=slice_tokenized_data, slice_idx=i, slice_partition_dirpath=slice_partition_dirpath))
          
    return slice_mapping

class ELMoTokenizedDatasetLarge(ELMoTokenizedDataset):
  
  def __init__(self, dataset: BaseDataset, type2id: dict, tokenizer, max_tokens, name: str, partition_name: str, slice_dirpath: str, slice_dimension: int = 10000, **kwargs) -> None:
    self.slice_manager = SliceManager(partition_name=partition_name, slice_dirpath=slice_dirpath, slice_dimension=slice_dimension, n_examples=dataset.get_elements_number())
    super().__init__(dataset, type2id, tokenizer, max_tokens, name, partition_name, **kwargs)

  def tokenize(self, original_sentences, processed_sentences, types):
    return self.slice_manager.tokenize(original_sentences=original_sentences, 
                                        processed_sentences=processed_sentences,
                                        types=types, 
                                        tokenize_fn=super().tokenize, 
                                        save_slice_fn=self.save_slice)

  def save_slice(self, slice_tokenized_data, slice_idx, slice_partition_dirpath):
    # prepare dirs
    tokenized_sentences_dirpath = os.path.join(slice_partition_dirpath,'tokenized_sentences')
    one_hot_types_dirpath = os.path.join(slice_partition_dirpath, 'one_hot_types')
    
    os.makedirs(os.path.join(tokenized_sentences_dirpath, 'input_ids'), exist_ok=True)
    os.makedirs(os.path.join(tokenized_sentences_dirpath, 'mention_mask'), exist_ok=True)
    
    os.makedirs(os.path.join(one_hot_types_dirpath), exist_ok=True)

    # prepare filenames
    slice_name = f'{slice_idx}.pt'
    slice_path = os.path.join(tokenized_sentences_dirpath, '{}', slice_name)
    slice_path_input_ids = slice_path.format('input_ids')
    slice_path_mention_mask = slice_path.format('mention_mask')
    
    slice_path_one_hot_types = os.path.join(one_hot_types_dirpath, slice_name)
    
    # save slice
    torch.save(slice_tokenized_data['tokenized_sentences']['input_ids'], slice_path_input_ids)
    torch.save(slice_tokenized_data['tokenized_sentences']['mention_mask'], slice_path_mention_mask)
    torch.save(slice_tokenized_data['one_hot_types'], slice_path_one_hot_types)

    # save mapping
    slice_mapping = {}
    slice_mapping[slice_idx] = {
      'tokenized_sentences': {
        'input_ids' : slice_path_input_ids,
        'mention_mask' : slice_path_mention_mask
      },
      'one_hot_types' : slice_path_one_hot_types
    }

    return slice_mapping


class BaseBERTTokenizedDatasetLarge(BaseBERTTokenizedDataset):
  def __init__(self, dataset: BaseDataset, type2id: dict, tokenizer, name: str, bertlike_model_name: str,
              max_mention_words: int = 5, max_left_words: int = 10, max_right_words: int = 10,
              max_tokens: int = 80, partition_name: str = '', slice_dirpath: str = '', slice_dimension: int = 10000):
    
    self.slice_manager = SliceManager(partition_name=partition_name, slice_dirpath=slice_dirpath, slice_dimension=slice_dimension, n_examples=dataset.get_elements_number())
    super().__init__(dataset, type2id, tokenizer, name, bertlike_model_name, max_mention_words, max_left_words, max_right_words, max_tokens, partition_name)


  def tokenize(self, original_sentences, processed_sentences, types):
    return self.slice_manager.tokenize(original_sentences=original_sentences, 
                                        processed_sentences=processed_sentences,
                                        types=types, 
                                        tokenize_fn=super().tokenize, 
                                        save_slice_fn=self.save_slice)

  def save_slice(self, slice_tokenized_data, slice_idx, slice_partition_dirpath):
    # prepare dirs
    tokenized_sentences_dirpath = os.path.join(slice_partition_dirpath,'tokenized_sentences')
    one_hot_types_dirpath = os.path.join(slice_partition_dirpath, 'one_hot_types')
    
    os.makedirs(os.path.join(tokenized_sentences_dirpath, 'input_ids'), exist_ok=True)
    os.makedirs(os.path.join(tokenized_sentences_dirpath, 'attention_mask'), exist_ok=True)
    
    os.makedirs(os.path.join(one_hot_types_dirpath), exist_ok=True)

    # prepare filenames
    slice_name = f'{slice_idx}.pt'
    slice_path = os.path.join(tokenized_sentences_dirpath, '{}', slice_name)
    slice_path_input_ids = slice_path.format('input_ids')
    slice_path_attention_mask = slice_path.format('attention_mask')
    
    slice_path_one_hot_types = os.path.join(one_hot_types_dirpath, slice_name)
    
    # save slice
    torch.save(slice_tokenized_data['tokenized_sentences']['input_ids'], slice_path_input_ids)
    torch.save(slice_tokenized_data['tokenized_sentences']['attention_mask'], slice_path_attention_mask)
    torch.save(slice_tokenized_data['one_hot_types'], slice_path_one_hot_types)

    # save mapping
    slice_mapping = {}
    slice_mapping[slice_idx] = {
      'tokenized_sentences': {
        'input_ids' : slice_path_input_ids,
        'attention_mask' : slice_path_attention_mask
      },
      'one_hot_types' : slice_path_one_hot_types
    }

    return slice_mapping