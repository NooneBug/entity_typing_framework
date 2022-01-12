from os import EX_SOFTWARE
from transformers import AutoModel
from pytorch_lightning.core.lightning import LightningModule
from transformers import PfeifferConfig, HoulsbyConfig

class BaseBERTLikeEncoder(LightningModule):
    def __init__(self, name: str, bertlike_model_name : str, freeze_encoder : bool = False) -> None:
        super().__init__()
        self.bertlike_model_name = bertlike_model_name
        self.encoder = AutoModel.from_pretrained(self.bertlike_model_name)
        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        self.freeze()

    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        raise Exception("Implement this function")

    def get_representation_dim(self):
        return self.encoder.config.dim

class DistilBERTEncoder(BaseBERTLikeEncoder):
    def __init__(self, bertlike_model_name: str = 'distilbert-base-uncased', **kwargs) -> None:
        super().__init__(bertlike_model_name=bertlike_model_name, **kwargs)
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        return self.aggregate_function(self.encoder(batched_tokenized_sentence, batched_attn_masks))

    def aggregate_function(self, encoder_output):
        return encoder_output['last_hidden_state'][:, 0, :]

class AdapterDistilBERTEncoder(DistilBERTEncoder):
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