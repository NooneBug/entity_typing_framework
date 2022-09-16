from torch.utils.data import Dataset
import numpy as np
import torch

class ET_DatasetLarge(Dataset):
   
  def __init__(self, tokenized_dataset):
    super().__init__()

    self.slice_mapping = tokenized_dataset.tokenized_data
    self.n_examples = len(tokenized_dataset)
    self.slice_dimension = tokenized_dataset.slice_manager.slice_dimension
  
  def get_slice_and_offset(self, idx):
    slice_idx = idx // self.slice_dimension
    offset = idx % self.slice_dimension
    return slice_idx, offset

  def get_tensor(self, slice_path, offset):
    return torch.load(slice_path)[offset, :]
  
  def unpack_item(self, slice_paths, offset):
    raise 'Implement this function'

  def __getitem__(self, idx):
    # get example's slice
    slice_idx, offset = self.get_slice_and_offset(idx)
    slice_paths = self.slice_mapping[slice_idx]

    # get item from slice
    item = self.unpack_item(slice_paths, offset)

    return item

  def __len__(self):
    return self.n_examples



class BERT_ET_DatasetLarge(ET_DatasetLarge):

  def unpack_item(self, slice_paths, offset):

    input_ids = self.get_tensor(slice_paths['tokenized_sentences']['input_ids'], offset)
    attention_mask = self.get_tensor(slice_paths['tokenized_sentences']['attention_mask'], offset)
    one_hot_types = self.get_tensor(slice_paths['one_hot_types'], offset)

    return input_ids, attention_mask, one_hot_types




class ELMo_ET_DatasetLarge(ET_DatasetLarge):
     
  def unpack_item(self, slice_paths, offset):
    tokenized_sentences = self.get_tensor(slice_paths['tokenized_sentences']['input_ids'], offset)
    mention_mask = self.get_tensor(slice_paths['tokenized_sentences']['mention_mask'], offset)
    one_hot_types = self.get_tensor(slice_paths['one_hot_types'], offset)

    return tokenized_sentences, mention_mask, one_hot_types
