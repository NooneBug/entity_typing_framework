from entity_typing_framework.dataset_classes.datasets import BaseDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

class BaseBERTTokenizedDataset(Dataset):
    def __init__(self,
                dataset : BaseDataset, 
                type2id : dict,
                name : str,
                bertlike_model_name : str, 
                max_mention_words : int = 5,
                max_left_words : int = 10,
                max_right_words : int = 10) -> None:
        super().__init__()
        
        self.dataset = dataset
        
        self.max_mention_words = max_mention_words
        self.max_left_words = max_left_words
        self.max_right_words = max_right_words
        
        self.tokenizer = AutoTokenizer.from_pretrained(bertlike_model_name)

        sentences = self.create_sentences_from_dataset(dataset)

        sentences = [self.create_sentence(s, self.max_mention_words, self.max_left_words, self.max_right_words) for s in sentences]

        self.tokenized_sentences, self.max_length, self.avg_length = self.tokenize(sentences)

        self.type2id = type2id
        
        self.tokenized_types = self.tokenize_types(dataset)
        self.one_hot_types = self.types2onehot(num_classes = len(self.type2id), types = self.tokenized_types)

    def types2onehot(self, num_classes, types):
        one_hot = torch.zeros(len(types), num_classes)
        for id, t in enumerate(types):
            one_hot[id, t] = 1
        return one_hot

    def tokenize_types(self, dataset):
        tokenized_types = [[self.type2id[t] for t in example_types] for example_types in dataset.labels]
        return tokenized_types
    
    def create_sentences_from_dataset(self, dataset):
        sentences = [
            {
                'mention_span' : dataset.mentions[i],
                'left_context_tokens' : dataset.left_contexts[i],
                'right_context_tokens' : dataset.right_contexts[i],
            }
            for i in range(dataset.get_elements_number())
            ]
        return sentences

    def create_sentence(self, sent_dict, max_mention_words = 5, max_left_words = 10, max_right_words = 10):
        return self.split_and_cut_mention(sent_dict['mention_span'], max_mention_words) + \
        ' [SEP] ' + self.cut_context(sent_dict['left_context_tokens'], max_left_words, False) + \
        ' [SEP] ' + self.cut_context(sent_dict['right_context_tokens'], max_right_words, True)
    
    def split_and_cut_mention(self, string, limit):
        if type(limit) == int:
            return ' '.join(string.split(' ')[:limit])
        else:
            return string

    def cut_context(self, string, limit, first):
        string = string.split(' ')
        if type(limit) == int:
            if limit:
                if first: 
                    return ' '.join(string[:limit])
                else:
                    return ' '.join(string[-limit:])
            else:
                return ''
        else:
            return ' '.join(string) 

    def tokenize(self, sentences):
        max_len, avg_len = self.compute_max_length(sentences)
        return self.tokenizer(sentences, return_tensors='pt', max_length = max_len, padding = 'max_length', truncation=True), max_len, avg_len
    
    def compute_max_length(self, sent):

        max_length = 0
        total_tokens = 0
        for s in tqdm(sent, desc="computing max length"):
            tokens = len(self.tokenizer(s, return_tensors='pt')['input_ids'][0])
            total_tokens += tokens
            if tokens > max_length:
                max_length = tokens
        avg_len = total_tokens/len(sent)
        print('\nmax length : {} (first and last tokens are [CLS] and [SEP])'.format(max_length))
        print('\navg length : {:.2f} (first and last tokens are [CLS] and [SEP])'.format(avg_len))

        return max_length, avg_len
        