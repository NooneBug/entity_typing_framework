from entity_typing_framework.dataset_classes.datasets import BaseDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class GeneralTokenizedDataset(Dataset):
    def __init__(self,
                dataset : BaseDataset, 
                type2id : dict,
                tokenizer,
                max_tokens,
                name : str) -> None:
        super().__init__()
        
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.sentences = self.extract_sentences_from_dataset(dataset)

        sentences = [self.create_sentence(s) for s in self.sentences]
        self.tokenized_sentences, self.max_length, self.avg_length = self.tokenize(sentences, self.max_tokens)

        self.type2id = type2id
        
        self.tokenized_types = self.tokenize_types(dataset)
        self.one_hot_types = self.types2onehot(num_types = len(self.type2id), types = self.tokenized_types)
    
    def types2onehot(self, num_types, types):
        '''
        tokenize id of types with one hot encoding

        parameters:
            num_types:
                the number of types in the dataset; automatically extracted by the :doc:`Dataset Manager <dataset_managers>`
            
            types:
                the list of types of each example in the dataset (not the list of all types, but the list of true types for each example in the dataset); automatically extracted by the :doc:`Dataset Reader <dataset_readers>` 
        
        return:
            a torch.tensor with shape :code:`len(types), num_types` containing one-hot encodings of types of each example in the dataset
        '''
        one_hot = torch.zeros(len(types), num_types)
        for id, t in enumerate(types):
            one_hot[id, t] = 1
        return one_hot
    
    def tokenize_types(self, dataset):
        '''
        tokenize alphanumerical types using ids in :code:`type2id`

        parameters:
            dataset:
                see the class parameters :code:`dataset`

        return:
            a list of list of ids corresponding to the true types of each example in the dataset
        '''
        tokenized_types = [[self.type2id[t] for t in example_types] for example_types in dataset.labels]
        return tokenized_types
    
    def extract_sentences_from_dataset(self, dataset):
        '''
        
        organizes the each sentence in a dictionary :code:`{mention_span : mention, left_context_token : list_of_left_context_tokens, right_context_token : list_of_right_context_tokens}` picking the attributed of :code:`dataset`

        parameters:
            dataset:
                see the class parameters :code:`dataset`

        return:
            a list of dictionaries, one dictionary for each sentence in the dataset

        '''
        sentences = [
            {
                'mention_span' : dataset.mentions[i],
                'left_context_tokens' : dataset.left_contexts[i],
                'right_context_tokens' : dataset.right_contexts[i],
            }
            for i in range(dataset.get_elements_number())
            ]
        return sentences
    
    def create_sentence(self, sentence):
        return f"{sentence['left_context_tokens']} {sentence['mention_span']} {sentence['right_context_tokens']}"

    def make_light(self):
        '''
        delete :code:`dataset` and :code:`tokenizer` from the object, this method is useful to save the BaseBERTTokenizedDataset in a light manner, avoiding the original dataset and the tokenizer
        '''
        
        self.dataset = None
        self.tokenizer = None
        self.sentences = None

    def tokenize(self, sentences, max_tokens=80):
        '''
        tokenizes all sentences using the tokenizer instantiate by :code:`instance_tokenizer`

        parameters:
            sentences:
                a list of sentences obtained by :code:`create_sentence` (repeatedly called)
            (optional) max_tokens:
                this param ensures that only the first :code:`max_tokens` tokens in each tokenized sentence are kept, the other tokens are discarded.

                accepted values are integers and the string :code:`"max"` to avoid the token discard
        
        return:
            see `BatchEncoding <https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/tokenizer#transformers.BatchEncoding>`_
        '''
        max_len, avg_len = self.compute_max_length(sentences, max_tokens) # JUAN
        # max_len = 52
        # avg_len = 29.31
        return self.tokenize_sentences(sentences, max_len), max_len, avg_len
    
    def compute_max_length(self, sent, max_tokens=80):
        '''
        compute the maximum number of tokens in the dataset. This method is used to set the :code:`max_length` parameters when calling the tokenizer.

        params:
            sent:
                see :code:`tokenize.sentences`
            
            (optional) max_tokens:
                see :code:`tokenize.max_tokens`
        '''

        max_length = 0
        total_tokens = 0
        for s in tqdm(sent, desc="computing max length"):
            tokens = min(len(self.tokenize_single_sentence(s)), max_tokens)
            total_tokens += tokens
            if tokens > max_length:
                max_length = tokens
        avg_len = total_tokens/len(sent)
        print('\nmax length : {} (special tokens may be included in the count)'.format(max_length))
        print('\navg length : {:.2f} (special tokens may be included in the count)'.format(avg_len))
        return max_length, avg_len
    
    def tokenize_single_sentence(self, sentence, max_tokens):
        raise Exception('Please, implement this function')

    def tokenize_sentences(self, sentences):
        raise Exception('Please, implement this function')

class ELMoTokenizedDataset(GeneralTokenizedDataset):

    def __init__(self, dataset: BaseDataset, type2id: dict, tokenizer, max_tokens, name: str) -> None:
        super().__init__(dataset, type2id, tokenizer, max_tokens, name)

        self.mention_mask = self.create_mention_mask()

    def create_mention_mask(self):
        mention_mask = torch.zeros((len(self.sentences), self.max_length), dtype=torch.int8)
        for i, s in enumerate(self.sentences):
            mention_start = len(s['left_context_tokens'].split())
            mention_end = mention_start + len(s['mention_span'].split())
            mention_mask[i][mention_start:mention_end] = 1
        return mention_mask

    def tokenize_sentences(self, sentences, max_len):
        # create a fake sentence to force the tokenizer pad the sentence to tokenize
        # assuming to use allennlp.modules.elmo.batch_to_ids
        return torch.stack([self.tokenizer([s, ['a' for _ in range(max_len)]])[0][:max_len,:] for s in tqdm(sentences, desc=f'tokenize sentences with max_lenght: {max_len}')])

    # def tokenize_sentences(self, max_len):
    #     # create a fake sentence to force the tokenizer pad the sentence to tokenize
    #     # assuming to use allennlp.modules.elmo.batch_to_ids
    #     tokenized_sentences = []
    #     fake_s = ['a' for _ in range(max_len)]
    #     for _ in tqdm(range(len(self.sentences)), desc=f'tokenize sentences with max_lenght: {max_len}'):
    #         s = self.sentences.pop(0)
    #         tokenized_sentences.append(self.tokenizer([s, fake_s])[0][:max_len,:])
    #     return torch.stack(tokenized_sentences)

    def tokenize_single_sentence(self, sentence):
        # assuming to use allennlp.modules.elmo.batch_to_ids
        return self.tokenizer(sentence)

    def create_sentence(self, sentence):
        return super().create_sentence(sentence).split()

class BaseBERTTokenizedDataset(GeneralTokenizedDataset):
    '''
    Tokenizes a dataset using `huggingface AutoTokenizer <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer>`_

    Parameters:
        dataset:
            the dataset loaded by a :doc:`Dataset Reader <dataset_readers>` to be tokenized

            Note : the format of the loaded dataset by the :doc:`Dataset Reader <dataset_readers>` has to be in accord with the one expected by :code:`create_sentences_from_dataset()`
        
        type2id:
            the :code:`type2id` dictionary created by the :doc:`Dataset Manager <dataset_managers>`
    
        name:
            the name of the submodule, has to be specified in the :code:`yaml` configuration file with key :code:`data.tokenizer_params.name`

            this param is used by the :doc:`Dataset Manager <dataset_managers>` to instance the correct submodule
        
        bertlike_model_name:
            required param, used to instance a `huggingface AutoTokenizer <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer>`_.
            
            this param drives the tokenizer used by the model. 

            this param has to be specified in the :code:`yaml` configuration file with key :code:`data.tokenizer_params.bertlike_model_name`. The value of this param has to be equal to the param :code:`model.network_params.encoder_params.bertlike_model_name`.
        
        (optional) max_mention_words:
            this param ensures that only the first :code:`max_mention_words` of each `entity mention` in the sentence are tokenized, the other words are discarded.

            accepted values are integers and the string :code:`"max"` to avoid the word discard
        
        (optional) max_left_words:
            this param ensures that only the last :code:`max_left_words` in each `left context` in a sentence are tokenized, the other words are discarded.

            accepted values are integers and the string :code:`"max"` to avoid the word discard

        (optional) max_right_words:
            this param ensures that only the first :code:`max_right_words` in each `right context` in a sentence are tokenized, the other words are discarded.

            accepted values are integers and the string :code:`"max"` to avoid the word discard

        (optional) max_tokens:
            this param ensures that only the first :code:`max_tokens` tokens in each tokenized sentence are kept, the other tokens are discarded.

            accepted values are integers and the string :code:`"max"` to avoid the token discard

    '''
    def __init__(self,
                dataset : BaseDataset, 
                type2id : dict,
                tokenizer, 
                name : str,
                bertlike_model_name : str,
                max_mention_words : int = 5,
                max_left_words : int = 10,
                max_right_words : int = 10,
                max_tokens : int = 80) -> None:


        self.max_mention_words = max_mention_words
        self.max_left_words = max_left_words
        self.max_right_words = max_right_words
        
        super().__init__(dataset, type2id, tokenizer, max_tokens, name)
        
        # self.dataset = dataset
        
        # self.tokenizer = tokenizer

        # sentences = self.extract_sentences_from_dataset(dataset)

        # sentences = [self.create_sentence(s) for s in sentences]

        # self.tokenized_sentences, self.max_length, self.avg_length = self.tokenize(sentences, self.max_tokens)

        # self.type2id = type2id
        
        # self.tokenized_types = self.tokenize_types(dataset)
        # self.one_hot_types = self.types2onehot(num_types = len(self.type2id), types = self.tokenized_types)

    
    # def types2onehot(self, num_types, types):
    #     '''
    #     tokenize id of types with one hot encoding

    #     parameters:
    #         num_types:
    #             the number of types in the dataset; automatically extracted by the :doc:`Dataset Manager <dataset_managers>`
            
    #         types:
    #             the list of types of each example in the dataset (not the list of all types, but the list of true types for each example in the dataset); automatically extracted by the :doc:`Dataset Reader <dataset_readers>` 
        
    #     return:
    #         a torch.tensor with shape :code:`len(types), num_types` containing one-hot encodings of types of each example in the dataset
    #     '''
    #     one_hot = torch.zeros(len(types), num_types)
    #     for id, t in enumerate(types):
    #         one_hot[id, t] = 1
    #     return one_hot

    # def tokenize_types(self, dataset):
    #     '''
    #     tokenize alphanumerical types using ids in :code:`type2id`

    #     parameters:
    #         dataset:
    #             see the class parameters :code:`dataset`

    #     return:
    #         a list of list of ids corresponding to the true types of each example in the dataset
    #     '''
    #     tokenized_types = [[self.type2id[t] for t in example_types] for example_types in dataset.labels]
    #     return tokenized_types
    
    # def extract_sentences_from_dataset(self, dataset):
    #     '''
        
    #     organizes the each sentence in a dictionary :code:`{mention_span : mention, left_context_token : list_of_left_context_tokens, right_context_token : list_of_right_context_tokens}` picking the attributed of :code:`dataset`

    #     parameters:
    #         dataset:
    #             see the class parameters :code:`dataset`

    #     return:
    #         a list of dictionaries, one dictionary for each sentence in the dataset

    #     '''
    #     sentences = [
    #         {
    #             'mention_span' : dataset.mentions[i],
    #             'left_context_tokens' : dataset.left_contexts[i],
    #             'right_context_tokens' : dataset.right_contexts[i],
    #         }
    #         for i in range(dataset.get_elements_number())
    #         ]
    #     return sentences

    def create_sentence(self, sent_dict):
        '''
        composes a sentence in the dataset in the format :code:`"mention [SEP] left_context_tokens [SEP] right_context_tokens"`. This method uses :code:`max_mention_words, max_left_words` and :code:`max_right_words` to discard words (using :code:`split_and_cut_mention` and :code:`cut_context`).

        parameters:
            sent_dict:
                a single dictionary extracted by :code:`extract_sentences_from_dataset`
        
        return :
            a sentence in the format :code:`"mention [SEP] left_context_tokens [SEP] right_context_tokens"`
        '''
        return self.split_and_cut_mention(sent_dict['mention_span'], self.max_mention_words) + \
        ' [SEP] ' + self.cut_context(sent_dict['left_context_tokens'], self.max_left_words, False) + \
        ' [SEP] ' + self.cut_context(sent_dict['right_context_tokens'], self.max_right_words, True)
    
    def split_and_cut_mention(self, mention, max_mention_words):
        '''
        split and cut the mention following the :code:`max_mention_words` limit

        parameters:
            mention:
                the string to be cutted
            
            max_mention_words:
                see class parameter :code:`max_mention_words`. if the value :code:`"max"` is passed, the entire :code:`mention` is maintained
        
        return:
            a string containing the mention
        '''
        if type(max_mention_words) == int:
            return ' '.join(mention.split(' ')[:max_mention_words])
        else:
            return mention

    def cut_context(self, context, limit, first):
        '''
        cut the left or the right context (depending on the value of :code:`first`)

        parameters:
            context:
                the string to be cutted
            
            limit:
                how many words maintain in :code:`context`. (commonly :code:`max_left_words` or :code:`max_right_words`)

                if the value :code:`"max"` is passed, the entire :code:`context` is maintained
            
            first:
                if :code:`True`, maintain the first :code:`limit` words. if :code:`False` maintain the last :code:`limit` words

        return:
            a string containing the context 

        '''
        string = context.split(' ')
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

    # def tokenize(self, sentences, max_tokens=80):
    #     '''
    #     tokenizes all sentences using the tokenizer instantiate by :code:`instance_tokenizer`

    #     parameters:
    #         sentences:
    #             a list of sentences obtained by :code:`create_sentence` (repeatedly called)
    #         (optional) max_tokens:
    #             this param ensures that only the first :code:`max_tokens` tokens in each tokenized sentence are kept, the other tokens are discarded.

    #             accepted values are integers and the string :code:`"max"` to avoid the token discard
        
    #     return:
    #         see `BatchEncoding <https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/tokenizer#transformers.BatchEncoding>`_
    #     '''
    #     max_len, avg_len = self.compute_max_length(sentences, max_tokens)
    #     return self.tokenize_sentences(sentences), max_len, avg_len
    
    def tokenize_sentences(self, sentences, max_len):
        return self.tokenizer(sentences, return_tensors='pt', max_length = max_len, padding = 'max_length', truncation=True)

    def tokenize_single_sentence(self, sentence):
        return self.tokenizer(sentence, return_tensors='pt')['input_ids'][0]

    # def compute_max_length(self, sent, max_tokens=80):
    #     '''
    #     compute the maximum number of tokens in the dataset. This method is used to set the :code:`max_length` parameters when calling the tokenizer.

    #     params:
    #         sent:
    #             see :code:`tokenize.sentences`
            
    #         (optional) max_tokens:
    #             see :code:`tokenize.max_tokens`
    #     '''

    #     max_length = 0
    #     total_tokens = 0
    #     for s in tqdm(sent, desc="computing max length"):
    #         tokens = min(len(self.tokenize_single_sentence(s)), max_tokens)
    #         total_tokens += tokens
    #         if tokens > max_length:
    #             max_length = tokens
    #     avg_len = total_tokens/len(sent)
    #     print('\nmax length : {} (first and last tokens are [CLS] and [SEP])'.format(max_length))
    #     print('\navg length : {:.2f} (first and last tokens are [CLS] and [SEP])'.format(avg_len))
    #     return max_length, avg_len
        
    # def make_light(self):
    #     '''
    #     delete :code:`dataset` and :code:`tokenizer` from the object, this method is useful to save the BaseBERTTokenizedDataset in a light manner, avoiding the original dataset and the tokenizer
    #     '''
        
    #     self.dataset = None
    #     self.tokenizer = None