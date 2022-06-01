from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embeddings_classes import CenterSigmoidBoxTensor
from pytorch_lightning.core.lightning import LightningModule


class OneHotTypeEncoder(LightningModule):
    '''
    since the DatasetManager commonly represents true labels as one hot, this is a dummy module, only returns the already correct labels

    parameters:
        name:
            the name of the submodule, has to be specified in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.type_encoder_params.name`

            to instance this type encoder insert the string :code:`OneHotTypeEncoder` in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.type_encoder_params.name`

            this param is used by the :ref:`Entity Typing Network<EntityTypingNetwork>` to instance the correct submodule
    '''
    def __init__(self, name, **kwargs):
        super().__init__()

    def forward(self, batched_labels):
        '''
        defines the forward pass of the type encoder. In the forwar the :code:`batched_labels` is returned as is since this is a dummy module

        Parameters:
            batched_labels:
                one of the batch elements returned by the :doc:`dataset<datasets>` (:code:`__getitem__` method). It contains one hot labels extracted from :doc:`Dataset Tokenizer<dataset_tokenizers>`

                the expected format of the one hot labels is a vector [label_number] with values 1 for each :code:`True` label, 0 for each :code:`False` label based on the :code:`type2id` dictionary into the :doc:`Dataset Manager<dataset_managers>`
        '''
        return batched_labels
    
    def get_state_dict(self, smart_save=True):
        return self.state_dict()

