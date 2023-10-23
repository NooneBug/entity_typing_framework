from typing import Any
from transformers import AutoModel, AutoModelForMaskedLM, BartForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
from transformers import PfeifferConfig, HoulsbyConfig
from torch.nn import LSTM, Dropout, Sigmoid, ReLU, Dropout, Softmax, GELU


class BaseBERTLikeEncoder(LightningModule):
    '''
    Base BERT-based encoder, instances a BERT-like model using `AutoModel <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoModel>`_
    
    Parameters:
        name: 
            the name of the submodule, has to be specified in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.encoder_params.name`
        
            this param is used by the :ref:`EntityTypingNetwork<EntityTypingNetwork>` to instance the correct submodule
        
        bertlike_model_name:
            required param, used to instance a `huggingface AutoModelr <https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoModel>`_.

            this param has to be specified in the :code:`yaml` configuration file with key :code:`model.network_params.encoder_params.bertlike_model_name`.

            this param drives the encoder used by the model. The value of this param has to be equal to the param :code:`data.tokenizer_params.bertlike_model_name`.
        
        freeze_encoder:
            optional param, if setted to True freeze the encoder specified in :code:`bertlike_model_name`

            this param has to be specified in the :code:`yaml` configuration file with key :code:`model.network_params.encoder_params.freeze_encoder`

    '''
    def __init__(self, name: str, bertlike_model_name : str, freeze_encoder : bool = False, 
                 is_mlm = False, is_bart = False) -> None:
        super().__init__()
        self.bertlike_model_name = bertlike_model_name
        if is_bart:
            self.encoder = BartForConditionalGeneration.from_pretrained(self.bertlike_model_name)
        if is_mlm:
            self.encoder = AutoModelForMaskedLM.from_pretrained(self.bertlike_model_name, output_hidden_states=True)
        else:
            self.encoder = AutoModel.from_pretrained(self.bertlike_model_name)
        if freeze_encoder:
            self.freeze_encoder()
            # unfreeze last n layers
            if type(freeze_encoder) == int:
                last_layers = self.encoder.encoder.layer[freeze_encoder:]
                for layer in last_layers:
                    for _, param in list(layer.named_parameters()):
                        param.requires_grad = True
    
    def freeze_encoder(self):
        '''
        Method used to freeze the encoder instanced with :code:`bertlike_model_name`
        '''
        self.freeze()

    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        '''
        define the forward pass of the encoder; by default is not implemented and raises an exception

        Parameters:
            batched_tokenized_sentence:
                one of the batch elements returned by the :doc:`dataset<datasets>` (:code:`__getitem__` method). It contains tokenized sentences extracted from :doc:`Dataset Tokenizer<dataset_tokenizers>`

            batched_attn_masks:
                one of the batch elements returned by the :doc:`dataset<datasets>` (:code:`__getitem__` method). It contains attention masks on the input sentence extracted from :doc:`Dataset Tokenizer<dataset_tokenizers>`

        '''
        raise Exception("Implement this function")

    def get_representation_dim(self):
        '''
        returns the output dim of the encoder

        Return:
            output_dim: dimension of the encoded input (e.g. 768 on DistilBERT)
        '''
        raise Exception("Implement this function")

    def get_state_dict(self, smart_save=True):
        return self.state_dict()

class BERTEncoder(BaseBERTLikeEncoder):
    '''
    Default class to instantiate BERT as encoder; subclass of :code:`BaseBERTLikeEncoder`.
    
    to instance this encoder insert the string :code:`BERTEncoder` in the :code:`yaml` configuration file under the key : :code:`model.ET_Network_params.encoder_params.name`
    
    Parameters:
        bertlike_model_name:
            see :code:`bertlike_model_name` in :code:`BaseBERTLikeEncoder`. In this class the default value is :code:`bert-base-uncased`
    '''
    def __init__(self, bertlike_model_name: str = 'bert-base-uncased', **kwargs) -> None:
        super().__init__(bertlike_model_name=bertlike_model_name, **kwargs)
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        '''
        defines the forward pass of the encoder. In the forward the :code:`batched_tokenized_sentence` is processed by DistilBERT and is returned using the method :code:`aggregate_function`
        
        Parameters:
            batched_tokenized_sentence:
                see :code:`batched_tokenized_sentence` in :code:`BaseBERTLikeEncoder`

            batched_attn_masks:
                see :code:`batched_attn_masks` in :code:`BaseBERTLikeEncoder`
        
        Return:
            encoder_output: parameters processed by DistilBERT and aggregated by the method :code:`aggregate_function`
        '''
        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)
        return self.aggregate_function(self.encoder(batched_tokenized_sentence, batched_attn_masks))

    def aggregate_function(self, encoder_output):
        '''
        aggregates the output of BERT by picking the CLS encoding

        Parameters:
            encoder_output:
                output of BERT, commonly shaped as [batch_size, token_number, 768]
        
        Return:
            cls_encoding: encoding of the first output token (CLS token), commonly shaped as [batch_size, 768] 

        '''
        return encoder_output['last_hidden_state'][:, 0, :]
    
    def get_representation_dim(self):
        '''
        returns the output dim of the encoder

        Return:
            output_dim: dimension of the encoded input (e.g. 768 on DistilBERT)
        '''
        return self.encoder.config.hidden_size
    
class DistilBERTEncoder(BaseBERTLikeEncoder):
    '''
    Default class to instantiate DistilBERT as encoder; subclass of :code:`BaseBERTLikeEncoder`.
    
    to instance this encoder insert the string :code:`DistilBERTEncoder` in the :code:`yaml` configuration file under the key : :code:`model.ET_Network_params.encoder_params.name`
    
    Parameters:
        bertlike_model_name:
            see :code:`bertlike_model_name` in :code:`BaseBERTLikeEncoder`. In this class the default value is :code:`distilbert-base-uncased`
    '''
    def __init__(self, bertlike_model_name: str = 'distilbert-base-uncased', **kwargs) -> None:
        super().__init__(bertlike_model_name=bertlike_model_name, **kwargs)
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        '''
        defines the forward pass of the encoder. In the forward the :code:`batched_tokenized_sentence` is processed by DistilBERT and is returned using the method :code:`aggregate_function`
        
        Parameters:
            batched_tokenized_sentence:
                see :code:`batched_tokenized_sentence` in :code:`BaseBERTLikeEncoder`

            batched_attn_masks:
                see :code:`batched_attn_masks` in :code:`BaseBERTLikeEncoder`
        
        Return:
            encoder_output: parameters processed by DistilBERT and aggregated by the method :code:`aggregate_function`
        '''
        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)
        return self.aggregate_function(self.encoder(batched_tokenized_sentence, batched_attn_masks))

    def aggregate_function(self, encoder_output):
        '''
        aggregates the output of DistilBERT by picking the CLS encoding

        Parameters:
            encoder_output:
                output of DistilBERT, commonly shaped as [batch_size, token_number, 768]
        
        Return:
            cls_encoding: encoding of the first output token (CLS token), commonly shaped as [batch_size, 768] 

        '''
        return encoder_output['last_hidden_state'][:, 0, :]
    
    def get_representation_dim(self):
        '''
        returns the output dim of the encoder

        Return:
            output_dim: dimension of the encoded input (e.g. 768 on DistilBERT)
        '''
        return self.encoder.config.dim


class AdapterDistilBERTEncoder(DistilBERTEncoder):
    '''
    Class to instance DistilBERT with Adapters; Subclass of :code:`DistilBERTEncoder`.

    to instance this encoder insert the string :code:`AdapterDistilBERTEncoder` in the :code:`yaml` configuration file under the key : :code:`model.ET_Network_params.encoder_params.name`

    Parameters:
        bertlike_model_name:
            see :code:`bertlike_model_name` in :code:`DistilBERTEncoder`.
        
        adapter_arch:
            adapter architecture from the original paper on adapters, accepted values are :code:`Pfeiffer` or :code:`Houlsby`. Other values will raise an exception

            this param can be specified in the :code:`yaml` configuration file with key :code:`model.network_params.encoder_params.adapter_arch`.
        
        reduction_factor:
            reduction factor from the original paper on adapters. Is an integer number used to setup the autoencoder-like architecture: the hidden units of the autoencoder are computed as :code:`transformer_size/reduction_factor`
        
        adapter_name:
            a name useful when multiple adapters are used.


    '''
    def __init__(self, bertlike_model_name: str = 'distilbert-base-uncased', 
                        adapter_arch = 'Pfeiffer', 
                        reduction_factor = 16, 
                        adapter_name = 'ET_adapter', 
                        **kwargs) -> None:
        super().__init__(bertlike_model_name=bertlike_model_name, **kwargs)

        if adapter_arch == 'Pfeiffer':
            conf = PfeifferConfig
        elif adapter_arch == 'Houlsby':
            conf = HoulsbyConfig
        else:
            raise Exception('Please provide a valid conf_arch value. {} given, accepted : {} '.format(adapter_arch, ['Pfeiffer, Houlsby']))
        
        self.encoder.add_adapter(adapter_name, conf(reduction_factor = reduction_factor))
        self.encoder.train_adapter(adapter_name)

    def get_state_dict(self, smart_save=True):
        if smart_save:
            state_dict = {k: v for k, v in self.state_dict().items() if 'adapters' in k}
        else:
            state_dict = self.state_dict()
        return state_dict



class AdapterBERTEncoder(BERTEncoder):
    '''
    Class to instance BERT with Adapters; Subclass of :code:`BERTEncoder`.

    to instance this encoder insert the string :code:`AdapterBERTEncoder` in the :code:`yaml` configuration file under the key : :code:`model.ET_Network_params.encoder_params.name`

    Parameters:
        bertlike_model_name:
            see :code:`bertlike_model_name` in :code:`BERTEncoder`.
        
        adapter_arch:
            adapter architecture from the original paper on adapters, accepted values are :code:`Pfeiffer` or :code:`Houlsby`. Other values will raise an exception

            this param can be specified in the :code:`yaml` configuration file with key :code:`model.network_params.encoder_params.adapter_arch`.
        
        reduction_factor:
            reduction factor from the original paper on adapters. Is an integer number used to setup the autoencoder-like architecture: the hidden units of the autoencoder are computed as :code:`transformer_size/reduction_factor`
        
        adapter_name:
            a name useful when multiple adapters are used.


    '''
    def __init__(self, bertlike_model_name: str = 'bert-base-uncased', 
                        adapter_arch = 'Pfeiffer', 
                        reduction_factor = 16, 
                        adapter_name = 'ET_adapter', 
                        **kwargs) -> None:
        super().__init__(bertlike_model_name=bertlike_model_name, **kwargs)

        if adapter_arch == 'Pfeiffer':
            conf = PfeifferConfig
        elif adapter_arch == 'Houlsby':
            conf = HoulsbyConfig
        else:
            raise Exception('Please provide a valid conf_arch value. {} given, accepted : {} '.format(adapter_arch, ['Pfeiffer, Houlsby']))
        
        self.encoder.add_adapter(adapter_name, conf(reduction_factor = reduction_factor))
        self.encoder.train_adapter(adapter_name)

    def get_state_dict(self, smart_save=True):
        if smart_save:
            state_dict = {k: v for k, v in self.state_dict().items() if 'adapters' in k}
        else:
            state_dict = self.state_dict()
        return state_dict

class GloVeEncoder(LightningModule):

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batched_tokenized_sentence, mention_mask, **kwargs):
        # mention average

        mention_vectors = batched_tokenized_sentence * mention_mask.unsqueeze(dim = 2).expand(-1, -1, batched_tokenized_sentence.shape[-1]) 
        
        num = torch.sum(mention_vectors, dim = 1)
        denom = torch.sum(mention_mask, dim = 1)

        
        averaged_mention = num/denom.unsqueeze(1).expand(-1, batched_tokenized_sentence.shape[-1])
        zeros = torch.zeros_like(averaged_mention)
        
        averaged_mention = torch.where(torch.isnan(averaged_mention), zeros, averaged_mention)

        return averaged_mention

    def get_representation_dim(self):
        return 300

    def get_state_dict(self, smart_save):
        return self.state_dict()

class LSTMGloVeEncoder(GloVeEncoder):
    def __init__(self, name, lstm_mention_window, **kwargs):
        super().__init__(name, **kwargs)
        self.lstm_mention_window = lstm_mention_window
        self.LSTM = LSTM(input_size = 300, hidden_size = 180, num_layers = 1, dropout = .5)
        self.glove_dropout = Dropout(.5)

    def forward(self, batched_tokenized_sentence, mention_mask, **kwargs):
        mention_representation = super().forward(batched_tokenized_sentence, mention_mask, **kwargs)

        extended_mention = self.extract_extended_mention(batched_tokenized_sentence=batched_tokenized_sentence, mention_mask=mention_mask)

        dropout_mention = self.glove_dropout(extended_mention)
        extended_mention_representation = self.lstm_dropout(self.LSTM(dropout_mention))
        return torch.hstack((mention_representation, extended_mention_representation))

    def extract_extended_mention(self, batched_tokenized_sentence, mention_mask, offset = 1):
        
        batched_tokenized_mention = torch.zeros_like(batched_tokenized_sentence)
        
        for i, x in enumerate(batched_tokenized_sentence):
            nz = mention_mask[i].nonzero()
            mention_len = len(nz)
            # TODO: better to avoid...
            if mention_len > 0:
                start_index = nz[0]
                end_index = start_index + mention_len
                batched_tokenized_mention[i, 0:mention_len + offset, :] = x[max(start_index - offset, 0) : end_index +1]
            else:
                print(f'Found an example with missing mention: try to set max_tokens higher than {batched_tokenized_sentence.shape[1]}')
        
        return batched_tokenized_mention

import torch.nn.functional as F

class NFETCEncoder(LSTMGloVeEncoder):
    def __init__(self, name, context_window, **kwargs):
        super().__init__(name, **kwargs)       
        self.context_window = context_window
        self.BiLSTM = LSTM(input_size = 300, hidden_size = 180, num_layers = 1, bidirectional = True, dropout = .5)

    def forward(self, batched_tokenized_sentence, mention_mask, **kwargs):
        mention_and_LSTM_repr = super().forward(batched_tokenized_sentence, mention_mask, **kwargs)

        mention_and_context = self.extract_extended_mention(batched_tokenized_sentence, mention_mask, offset=self.context_window)

        dropout_mention = self.glove_dropout(mention_and_context)
        extended_mention_representation = self.apply_attention(self.BiLSTM(dropout_mention))

        return torch.hstack((mention_and_LSTM_repr, extended_mention_representation))

        
    def apply_attention(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        #M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden

from allennlp.modules.elmo import Elmo
import torch
from time import time
ELMO_ENCODING_MODES = ['mmc', 'm_mc']
class ELMoEncoder(LightningModule):
    def __init__(self, name, option_file_path, weight_file_path, encoding_mode='mmc', *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        print('Instantiating ELMo Encoder, it requires a lot of time')
        t = time()

        self.encoder = Elmo(options_file=option_file_path, 
                            weight_file=weight_file_path, 
                            num_output_representations = 1)

        self.encoding_mode = encoding_mode
        # NOTE: use only for debugging
        # self.encoder = torch.load(option_file_path.replace('options.json', 'elmo.pt'))
        elapsed_time = time() - t
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("ELMo Encoder loaded in {:0>2}h{:0>2}m{:05.2f}s".format(int(hours),int(minutes),seconds))


    def forward(self, batched_tokenized_sentence, mention_mask, *args, **kwargs) -> Any:

        # batched_tokenized_sentence : [batch, max_len, 50]
        # mention_mask : [batch, max_len]

        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)

        elmo_representation = self.encoder(batched_tokenized_sentence)['elmo_representations'][0]
        # elmo_representation : [batch, max_len, 1024]

        # extract mc representation
        mc_representation = torch.max(elmo_representation, dim=1)[0]
        # mc_representation : [batch, 1024]

        if self.encoding_mode == 'mmc':
            # prepare mask to filter mention representation influenced by context
            mention_mask_unsqueezed = mention_mask.unsqueeze(2)
            shape = elmo_representation.shape
            mention_mask_expanded = mention_mask_unsqueezed.expand(*shape)
            # extract mention_representation
            mention_representation = torch.max(mention_mask_expanded * elmo_representation, dim=1)[0]
        elif self.encoding_mode == 'm_mc':
            # prepare isolated mentions placed at the beginning of the input
            batched_tokenized_mention = torch.zeros_like(batched_tokenized_sentence)
            for i, x in enumerate(batched_tokenized_sentence):
                nz = mention_mask[i].nonzero()
                mention_len = len(nz)
                # TODO: better to avoid...
                if mention_len > 0:
                    start_index = nz[0]
                    end_index = start_index + mention_len
                    batched_tokenized_mention[i, 0:mention_len, :] = x[start_index:end_index]
                else:
                    print(f'Found an example with missing mention: try to set max_tokens higher than {batched_tokenized_sentence.shape[1]}')
            # extract mention representation
            mention_elmo_representation = self.encoder(batched_tokenized_mention)['elmo_representations'][0]
            mention_representation = torch.max(mention_elmo_representation, dim=1)[0]
            # mention_representation : [batch, 1024]
        else:
            raise Exception(f"Please provide a valid encoding_mode value. '{self.encoding_mode}' given, accepted : {ELMO_ENCODING_MODES} ")

        # concatanate mention and mc representations
        mention_mc_representation = torch.hstack([mention_representation, mc_representation])

        return mention_mc_representation

    def get_representation_dim(self):
        # 2048 = 1024 (mention) + 1024 (mention_context)
        return 2048

    def get_state_dict(self, smart_save=True):
        return self.state_dict()

# IMPLEMENTED_POOLING_FUNCTIONS = ['min', 'max']

# class ELMoEncoder(LightningModule):
    
#     def __init__(self, option_file_path, weight_file_path, return_format, pooling_function = 'max', *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)

#         self.return_format = return_format
#         self.pooling_function_name = pooling_function

#         if pooling_function == 'max':
#             self.pooling_function = torch.max
#         elif pooling_function == 'min':
#             self.pooling_function = torch.min
#         else:
#             raise Exception(f'Please provide an implemented pooling_function name ({IMPLEMENTED_POOLING_FUNCTIONS}) or implement the given one {pooling_function}')

        # print('Instantiating ELMo Encoder, it require a lot of time')
        # t = time()
        
#         self.encoder = _ElmoBiLm(options_file=option_file_path,
#                                 weight_file=weight_file_path)
        
        # elapsed_time = time() - t
        # hours, rem = divmod(elapsed_time, 3600)
        # minutes, seconds = divmod(rem, 60)
        # print("ELMo Encoder loaded in {:0>2}h{:0>2}m{:05.2f}s".format(int(hours),int(minutes),seconds))
    
#     def extract_char_embedding(self, encoded_input):
#         # return char embeddings excluding first and last chars (in ELMo BOS and EOS)
#         return encoded_input[:, 1:, :]

#     def pool_contextualized_embedding(self, encoded_input):
#         # pool along the number of word in the sentence
#         # max or min pooling can handle padding operated by ELMo tokenizer
#         return self.pooling_function(encoded_input, dim = 1)
        
#     def forward(self, inputs) -> Any:
#         encoded_input = self.encoder(inputs)['activations']
        
#         if type(self.return_format) == int and self.return_format < 3 and self.return_format >= 0:
            
#             # extract the embedding of interest
#             filtered_encoded_input = encoded_input[self.return_format]
            
#             if self.return_format == 0:
#                 return self.extract_char_embedding(filtered_encoded_input)

#             elif self.return_format == 1 or self.return_format == 2:
#                 return self.pool_contextualized_embedding(filtered_encoded_input)

#         elif self.return_format == 'concat':
#             char_embedding = self.extract_char_embedding(encoded_input[0])
#             lvl1_embedding = self.pool_contextualized_embedding(encoded_input[1])
#             lvl2_embedding = self.pool_contextualized_embedding(encoded_input[2])

#             return torch.stack()

class MLMBERTEncoder(BERTEncoder):
    def __init__(self, mask_token_id, bertlike_model_name: str = 'bert-base-uncased', **kwargs) -> None:
        super().__init__(bertlike_model_name, is_mlm = True, **kwargs)
        self.mask_token_id = mask_token_id
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):

        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)

        mlm_output = self.encoder(batched_tokenized_sentence, batched_attn_masks)['logits']

        input_shape = mlm_output.shape

        masked_ids = torch.nonzero(batched_tokenized_sentence == self.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1) 

        masked_ids = masked_ids.repeat(1, 1, input_shape[2]).reshape(input_shape[0], 1, input_shape[2])
        score = torch.gather(mlm_output, 1, masked_ids) # batch x 1 x vocab_size 

        return score


class BARTEncoder(BERTEncoder):
    def __init__(self, bertlike_model_name: str = 'facebook/bart-base', **kwargs) -> None:
        super().__init__(bertlike_model_name, is_bart = True, **kwargs)
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):

        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)

        mlm_output = self.encoder(batched_tokenized_sentence, batched_attn_masks)

class PROMETBERTEncoder(BERTEncoder):
    def __init__(self, mask_token_id, bertlike_model_name: str = 'bert-base-uncased', embedding_dropout=None, activation_name='none', 
                 is_mlm=False, is_bart=False, **kwargs) -> None:
        super().__init__(bertlike_model_name, is_mlm = is_mlm, is_bart=is_bart,**kwargs)
        self.mask_token_id = mask_token_id

        if embedding_dropout:
            self.dropout = Dropout(float(embedding_dropout))
        else:
            self.dropout = None

        if activation_name == 'relu':
            self.activation = ReLU()
        elif activation_name == 'sigmoid':
            self.activation = Sigmoid()
        elif activation_name == 'softmax':
            self.activation = Softmax(dim=1)
        elif activation_name == 'gelu':
            self.activation = GELU()
        elif activation_name == 'none':
            self.activation = None
        else:
            raise Exception('An unknown name (\'{}\')is given for activation, check the yaml or implement an activation that correspond to that name'.format(activation_name))
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):

        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)

        mlm_output = self.encoder(batched_tokenized_sentence, batched_attn_masks)['last_hidden_state']

        input_shape = mlm_output.shape

        masked_ids = torch.nonzero(batched_tokenized_sentence == self.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1) 

        masked_ids = masked_ids.repeat(1, 1, input_shape[2]).reshape(input_shape[0], 1, input_shape[2])
        score = torch.gather(mlm_output, 1, masked_ids) # batch x 1 x encoder_dim

        if self.activation:
            score = self.activation(score)

        if self.dropout:
            score = self.dropout(score)

        return score

class PROMETAdapterBERTEncoder(AdapterBERTEncoder):
    def __init__(self, mask_token_id, bertlike_model_name: str = 'bert-base-uncased', embedding_dropout=None, activation_name='none', 
                 is_mlm = False, is_bart=False,**kwargs) -> None:
        super().__init__(bertlike_model_name, is_mlm=is_mlm, is_bart=is_bart,**kwargs)
        self.mask_token_id = mask_token_id

        if embedding_dropout:
            self.dropout = Dropout(float(embedding_dropout))
        else:
            self.dropout = None

        if activation_name == 'relu':
            self.activation = ReLU()
        elif activation_name == 'sigmoid':
            self.activation = Sigmoid()
        elif activation_name == 'softmax':
            self.activation = Softmax(dim=1)
        elif activation_name == 'gelu':
            self.activation = GELU()
        elif activation_name == 'none':
            self.activation = None
        else:
            raise Exception('An unknown name (\'{}\')is given for activation, check the yaml or implement an activation that correspond to that name'.format(activation_name))
        
    def forward(self, batched_tokenized_sentence, batched_attn_masks):

        batched_tokenized_sentence = batched_tokenized_sentence.to(torch.int32)
        batched_attn_masks = batched_attn_masks.to(torch.int32)

        mlm_output = self.encoder(batched_tokenized_sentence, batched_attn_masks)['last_hidden_state']

        input_shape = mlm_output.shape

        masked_ids = torch.nonzero(batched_tokenized_sentence == self.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1) 

        masked_ids = masked_ids.repeat(1, 1, input_shape[2]).reshape(input_shape[0], 1, input_shape[2])
        score = torch.gather(mlm_output, 1, masked_ids) # batch x 1 x encoder_dim

        if self.activation:
            score = self.activation(score)
            
        if self.dropout:
            score = self.dropout(score)

        return score
    
class PROMETBARTEncoder(PROMETBERTEncoder):
    def __init__(self, mask_token_id, bertlike_model_name: str = 'facebook/bart-base', embedding_dropout=None, activation_name='none', is_mlm=False, is_bart=True, **kwargs) -> None:
        super().__init__(mask_token_id, bertlike_model_name, embedding_dropout, activation_name, is_mlm, is_bart, **kwargs)

class PROMETAdapterBARTEncoder(PROMETAdapterBERTEncoder):
    def __init__(self, mask_token_id, bertlike_model_name: str = 'facebook/bart-base', embedding_dropout=None, activation_name='none', is_mlm=False, is_bart=True, **kwargs) -> None:
        super().__init__(mask_token_id, bertlike_model_name, embedding_dropout, activation_name, is_mlm, is_bart, **kwargs)