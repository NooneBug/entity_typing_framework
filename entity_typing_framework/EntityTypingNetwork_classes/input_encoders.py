from transformers import AutoModel
from pytorch_lightning.core.lightning import LightningModule
from transformers import PfeifferConfig, HoulsbyConfig

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
    def __init__(self, name: str, bertlike_model_name : str, freeze_encoder : bool = False) -> None:
        super().__init__()
        self.bertlike_model_name = bertlike_model_name
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