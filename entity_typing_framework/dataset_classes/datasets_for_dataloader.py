from torch.utils.data import Dataset
import numpy as np

class ET_Dataset(Dataset):

    def __init__(self, tokenized_dataset):
        super().__init__()

        self.tokenized_sentences = np.asarray(tokenized_dataset.tokenized_sentences['input_ids'])
        self.attn_masks = np.asarray(tokenized_dataset.tokenized_sentences['attention_mask'])
        self.one_hot_types = np.asarray(tokenized_dataset.one_hot_types)
    
    def __getitem__(self, idx):
        return self.tokenized_sentences[idx], self.attn_masks[idx], self.one_hot_types[idx]

    def __len__(self):
        return len(self.tokenized_sentences)
