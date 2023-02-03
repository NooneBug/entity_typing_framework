from entity_typing_framework.EntityTypingNetwork_classes.projectors import Layer, Projector
import torch.nn as nn
from torch.nn import ModuleDict, Sigmoid
import torch
from typing import Tuple
from pytorch_lightning.core.lightning import LightningModule

class Type2VecProjector(Projector):
  def __init__(self, embeddings_path, layers_parameters, check_parameters=True, **kwargs) -> None:
    super().__init__(**kwargs)
    # prepare type2vec vectors
    self.vector_decoder = VectorDecoder(embeddings_path)
    # prepare projection network
    self.layers_parameters = layers_parameters
    self.add_parameters()
    if check_parameters:
      self.check_parameters()
    self.layers = ModuleDict({layer_name: Layer(**layer_parameters) for layer_name, layer_parameters in self.layers_parameters.items()})     

  # TODO documentation
  def forward(self, input_representation):
    projected_input = self.project_input(input_representation)
    similarities = self.classify(projected_input)
    return similarities
  
  # TODO documentation
  def project_input(self, input_representation):
    # iteratively forward for each layer
    for i in range(len(self.layers_parameters)):
      if i == 0:
        h = self.layers[str(i)](input_representation)
      else:
        h = self.layers[str(i)](h)
    
    return h

  # TODO documentation
  def classify(self, projected_input):
    return self.vector_decoder(projected_input)

  def add_parameters(self):
    '''
    adds the default parameters if are not specified into the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.input_projector_params.layers_parameters`

    The default values are: 
        - if input features of the 0th projection layer are not specified or it is specified with the string :code:`encoder_dim`, the value :code:`input_dim` is inserted by default
        - if input features of a projection layer is specified with the string :code:`previous_out_features` the value :code:`out_features` of the previous layer is inserted
        - if output features of the last proection layer is specified with the string :code:`embedding_dim`: the value :code:`embedding_dim` is inserted by default
    '''
    if 'in_features' not in self.layers_parameters['0']:
      self.layers_parameters['0']['in_features'] = self.input_dim
    
    if self.layers_parameters['0']['in_features'] == 'encoder_dim':
      self.layers_parameters['0']['in_features'] = self.input_dim
    
    if 'out_features' not in self.layers_parameters[str(len(self.layers_parameters) - 1)]:
      self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.vector_decoder.vector_embeddings.embedding_dim
    
    if self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] == 'embedding_dim':
      self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.vector_decoder.vector_embeddings.embedding_dim

    previous_out_features = self.input_dim
    for k in self.layers_parameters:
      if self.layers_parameters[k]['out_features'] == 'in_features':
        self.layers_parameters[k]['out_features'] = self.layers_parameters[k]['in_features']
      if self.layers_parameters[k]['in_features'] == 'previous_out_features':
        self.layers_parameters[k]['in_features'] = previous_out_features
      previous_out_features = self.layers_parameters[k]['out_features']

  def check_parameters(self):
    '''
    Check the parameters values and raises exceptions. Ensure that a classic type2vec projector can be obtained.
    '''
    if self.input_dim != self.layers_parameters['0']['in_features']:
      raise Exception('Encoder\'s output dimension ({}) and projector\'s input dimension ({}) has to have the same value ({}). Check the yaml'.format(self.input_dim, self.layers_parameters['0']['in_features'], self.input_dim))
    
    if self.vector_decoder.vector_embeddings.embedding_dim != self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features']:
      raise Exception('Embedding\'s dimension ({}) and projector\'s last layer output dimension ({}) has to have the same value ({}). Check the yaml'.format(self.vector_decoder.vector_embeddings.embedding_dim, self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'], self.type_number))


class VectorDecoder(LightningModule):
  def __init__(self, embedding_path):
    super(VectorDecoder, self).__init__()
    # self.sig = Sigmoid()
    # load and freeze weights
    weights = torch.load(embedding_path)  
    self.vector_embeddings = nn.Embedding.from_pretrained(weights, freeze=False)
  
  def forward(
    self,
    mc_vector: torch.Tensor) -> Tuple[torch.Tensor, None]:
    inputs = torch.arange(0,
                          self.vector_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.vector_embeddings.weight.device)
    type_emb = self.vector_embeddings(inputs)

    # return mc_vector@type_emb.T
    return (self.sim_matrix(mc_vector, type_emb) + 1) / 2 # LabelRankingLoss
    # return mc_vector, type_emb # CosineEmbeddingLoss
    # return torch.cdist(mc_vector, type_emb)

  def get_state_dict(self, smart_save=True):
    return self.state_dict()


  def sim_matrix(self, a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt