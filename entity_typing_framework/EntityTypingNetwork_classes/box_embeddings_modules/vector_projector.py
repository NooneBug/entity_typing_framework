from typing import Optional
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Projector, ProjectorForIncrementalTraining
import torch.nn as nn
import torch
from typing import Optional, Tuple
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from copy import deepcopy

class VectorEmbeddingProjector(Projector):
    def __init__(self, type_number, input_dim, projection_network_params, vector_embeddings_dimension=218, **kwargs) -> None:
        super().__init__(input_dim=input_dim, type_number=type_number, **kwargs)
        
        self.vector_embeddings_dimension = vector_embeddings_dimension
        self.projection_network = HighwayNetwork(input_dim = input_dim,
                                                output_dim=self.vector_embeddings_dimension,
                                                **projection_network_params)
        self.vector_decoder = VectorDecoder(num_embeddings = type_number, embedding_dim = 218)

    def forward(self, encoded_input):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        mention_context_rep = self.projection_network(encoded_input)

        probs = self.vector_decoder(mention_context_rep)

        return probs
    
    # TODO documentation
    def project_input(self, input_representation):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        mention_context_rep = self.projection_network(input_representation)

        return mention_context_rep

    # TODO documentation
    def classify(self, projected_input):
      return self.vector_decoder(projected_input)

class VectorEmbeddingIncrementalProjector(ProjectorForIncrementalTraining):
    def get_class_for_pretrained_projector(self):
      return VectorEmbeddingProjector
    
    def get_class_for_incremental_projector(self):
      return VectorEmbeddingProjector
    
    def get_kwargs_incremental_projector(self, **kwargs):

        new_type_number = self.get_new_type_number(**kwargs)

        kwargs_additional_classifiers = deepcopy(kwargs)
        kwargs_additional_classifiers['type_number'] = new_type_number

        return kwargs_additional_classifiers
    
    def copy_pretrained_parameters_into_incremental_module(self):
      # assuming that pretrained_projector and additional_projector have the same architecture

      # copy nonlinear
      for pretrained_l, incremental_l in zip(self.pretrained_projector.projection_network.nonlinear, 
                                                self.additional_projector.projection_network.nonlinear):
            incremental_l.weight = torch.nn.Parameter(pretrained_l.weight.detach().clone())
            incremental_l.bias = torch.nn.Parameter(pretrained_l.bias.detach().clone())

      # copy gate
      for pretrained_l, incremental_l in zip(self.pretrained_projector.projection_network.gate, 
                                                self.additional_projector.projection_network.gate):
            incremental_l.weight = torch.nn.Parameter(pretrained_l.weight.detach().clone())
            incremental_l.bias = torch.nn.Parameter(pretrained_l.bias.detach().clone())
      
      # copy final_linear_layer
      self.additional_projector.projection_network.final_linear_layer.weight = torch.nn.Parameter(self.pretrained_projector.projection_network.final_linear_layer.weight.detach().clone())
      self.additional_projector.projection_network.final_linear_layer.bias = torch.nn.Parameter(self.pretrained_projector.projection_network.final_linear_layer.bias.detach().clone())

class VectorDecoder(LightningModule):
  def __init__(self,
               num_embeddings: int,
               embedding_dim: int = 218):
    super(VectorDecoder, self).__init__()

    self.num_embeddings = num_embeddings
    self.vector_embedding_dim = embedding_dim
    self.vector_embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim)
    self.sig = nn.Sigmoid()
  
  def forward(
    self,
    mc_vector: torch.Tensor) -> Tuple[torch.Tensor, None]:
    inputs = torch.arange(0,
                          self.vector_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.vector_embeddings.weight.device)
    type_emb = self.vector_embeddings(inputs)  # num types x 2*vector_embedding_dim
    
    probs = torch.matmul(mc_vector, type_emb.T)

    return self.sig(probs)

  def get_state_dict(self, smart_save=True):
        return self.state_dict()


class HighwayNetwork(LightningModule):
  def __init__(self,
               name,
               input_dim: int,
               output_dim: int,
               n_layers: int = 2,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)

