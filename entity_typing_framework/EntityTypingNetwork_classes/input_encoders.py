from transformers import AutoModel
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Linear

class BaseBERTLikeEncoder(LightningModule):
    def __init__(self, bertlike_model_name : str) -> None:
        super().__init__()
        self.bertlike_model_name = bertlike_model_name
        self.encoder = AutoModel.from_pretrained(self.bertlike_model_name)

    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        raise Exception("Implement this function")

    def get_representation_dim(self):
        return self.encoder.config.dim

class DistilBERTEncoder(BaseBERTLikeEncoder):
    def __init__(self, bertlike_model_name: str = 'distilbert-base-uncased') -> None:
        super().__init__(bertlike_model_name=bertlike_model_name)
    
    def forward(self, batched_tokenized_sentence, batched_attn_masks):
        return self.aggregate_function(self.encoder(batched_tokenized_sentence, batched_attn_masks))

    def aggregate_function(self, encoder_output):
        return encoder_output['last_hidden_state'][:, 0, :]