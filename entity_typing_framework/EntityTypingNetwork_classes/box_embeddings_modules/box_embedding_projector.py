from typing import Optional
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embeddings_classes import CenterSigmoidBoxTensor
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Projector, ProjectorForIncrementalTraining
import torch.nn as nn
import torch
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embeddings_classes import CenterSigmoidBoxTensor, BoxTensor, log1mexp
from typing import Optional, Tuple
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from copy import deepcopy

box_types = {
'CenterSigmoidBoxTensor': CenterSigmoidBoxTensor
}

class BoxEmbeddingProjector(Projector):
    def __init__(self, type_number, input_dim, box_decoder_params, projection_network_params, box_embeddings_dimension=109, **kwargs) -> None:
        super().__init__(input_dim=input_dim, type_number=type_number, **kwargs)
        
        self.box_embedding_dimension = box_embeddings_dimension
        self.projection_network = HighwayNetwork(input_dim = input_dim,
                                                output_dim=self.box_embedding_dimension * 2,
                                                **projection_network_params)
        self.mc_box = CenterSigmoidBoxTensor
        self.box_decoder = BoxDecoder(num_embeddings = type_number, embedding_dim = self.box_embedding_dimension, **box_decoder_params)

    def forward(self, encoded_input):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        projected_input = self.projection_network(encoded_input)

        # assuming that Boxes are CenterSigmoidBoxTensor, split the projected_input (a single tensor for each batch element) into two tensors for each batch elements to represent the box 
        mention_context_rep = self.mc_box.from_split(projected_input)

        log_probs = self.box_decoder(mention_context_rep)

        return mention_context_rep, log_probs
    
    # TODO documentation
    def project_input(self, input_representation):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        projected_input = self.projection_network(input_representation)

        # assuming that Boxes are CenterSigmoidBoxTensor, split the projected_input (a single tensor for each batch element) into two tensors for each batch elements to represent the box 
        mention_context_rep = self.mc_box.from_split(projected_input)

        return mention_context_rep

    # TODO documentation
    def classify(self, projected_input):
      return self.box_decoder(projected_input)

class BoxEmbeddingIncrementalProjector(ProjectorForIncrementalTraining):
    def get_class_for_pretrained_projector(self):
      return BoxEmbeddingProjector
    
    def get_class_for_incremental_projector(self):
      return BoxEmbeddingProjector
    
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

      # copy BoxDecoder weights
      # init new parameters to better exploit the hierarchy: the weights of the embedding of a new type are set to the values of the father's ones]
      pretrained_box_decoder = self.pretrained_projector.box_decoder.box_embeddings
      additional_box_decoder = self.additional_projector.box_decoder.box_embeddings
      for t in self.new_types:
          father = '/'.join(t.split('/')[:-1])
          idx_father = self.pretrained_projector.type2id[father]
          idx_t = self.additional_projector.type2id[t] - self.pretrained_projector.type_number
          additional_box_decoder.weight.data[idx_t] = torch.nn.Parameter(pretrained_box_decoder.weight[idx_father].detach().clone())

class BoxDecoder(LightningModule):
  def __init__(self,
               num_embeddings: int,
               embedding_dim: int = 109,
               box_type: str = 'CenterSigmoidBoxTensor',
               padding_idx: Optional[int] = None,
               max_norm: Optional[float] = None,
               norm_type: float = 2.,
               scale_grad_by_freq: bool = False,
               sparse: bool = False,
               _weight: Optional[torch.Tensor] = None,
               init_interval_delta: float = 0.5,
               init_interval_center: float = 0.01,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               n_negatives: int = 0,
               neg_temp: float = 0.,
               box_offset: float = 0.5,
               pretrained_box: Optional[torch.Tensor] = None,
               use_gumbel_baysian: bool = False,
               gumbel_beta: float = 1.0):
    super(BoxDecoder, self).__init__()

    self.num_embeddings = num_embeddings
    self.box_embedding_dim = embedding_dim
    self.box_type = box_type
    try:
      self.box = box_types[box_type]
    except KeyError as ke:
      raise ValueError("Invalid box type {}".format(box_type)) from ke
    self.box_offset = box_offset  # Used for constant tensor
    self.init_interval_delta = init_interval_delta
    self.init_interval_center = init_interval_center
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.n_negatives = n_negatives
    self.neg_temp = neg_temp
    self.use_gumbel_baysian = use_gumbel_baysian
    self.gumbel_beta = gumbel_beta
    self.box_embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)

    self.euler_gamma = 0.57721566490153286060

    if pretrained_box is not None:
      print('Init box emb with pretrained boxes.')
      print(self.box_embeddings.weight)
      self.box_embeddings.weight = nn.Parameter(pretrained_box)
      print(self.box_embeddings.weight)
  
  def _compute_hard_min_max(
    box1: BoxTensor,
    box2: BoxTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.max(box1.z, box2.z)
    max_point = torch.min(box1.Z, box2.Z)
    return min_point, max_point
    
  def _compute_gumbel_min_max(
    box1: BoxTensor,
    box2: BoxTensor,
    gumbel_beta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.stack([box1.z, box2.z])
    min_point = torch.max(
        gumbel_beta * torch.logsumexp(min_point / gumbel_beta, 0),
        torch.max(min_point, 0)[0])

    max_point = torch.stack([box1.Z, box2.Z])
    max_point = torch.min(
        -gumbel_beta * torch.logsumexp(-max_point / gumbel_beta, 0),
        torch.min(max_point, 0)[0])
    return min_point, max_point
  
  def init_weights(self):
    print('before', self.box_embeddings.weight)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., :self.box_embedding_dim],
      -self.init_interval_center, self.init_interval_center)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., self.box_embedding_dim:],
      self.init_interval_delta, self.init_interval_delta)
    print('after', self.box_embeddings.weight)

  def log_soft_volume(
    self,
    z: torch.Tensor,
    Z: torch.Tensor,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta: float = 0.) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if gumbel_beta <= 0.:
      return (torch.sum(
        torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)),
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
      return (torch.sum(
        torch.log(
          F.softplus(Z - z - 2 * self.euler_gamma * gumbel_beta, beta=temp).clamp_min(
          # F.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma, beta=temp).clamp_min(
            eps)),
          dim=-1) + torch.log(s))

  def type_box_volume(self) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    vol = self.log_soft_volume(type_box.z,
                               type_box.Z,
                               temp=self.inv_softplus_temp,
                               scale=self.softplus_scale,
                               gumbel_beta=self.gumbel_beta)
    return vol

  def get_pairwise_conditional_prob(self,
                                    type_x_ids: torch.Tensor,
                                    type_y_ids: torch.Tensor) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    type_x = emb[type_x_ids]
    type_y = emb[type_y_ids]
    type_x_box = self.box.from_split(type_x)
    type_y_box = self.box.from_split(type_y)

    # Compute intersection volume
    if self.use_gumbel_baysian:
      # Gumbel intersection
      min_point, max_point = self._compute_gumbel_min_max(type_x_box,
                                                     type_y_box,
                                                     self.gumbel_beta)
    else:
      min_point, max_point = self._compute_hard_min_max(type_x_box, type_y_box)

    intersection_vol = self.log_soft_volume(min_point,
                                            max_point,
                                            temp=self.inv_softplus_temp,
                                            scale=self.softplus_scale,
                                            gumbel_beta=self.gumbel_beta)
    # Compute y volume here
    y_vol = self.log_soft_volume(type_y_box.z,
                                 type_y_box.Z,
                                 temp=self.inv_softplus_temp,
                                 scale=self.softplus_scale,
                                 gumbel_beta=self.gumbel_beta)

    # Need to be careful about numerical issues
    conditional_prob = intersection_vol - y_vol
    return torch.cat([conditional_prob.unsqueeze(-1),
                      log1mexp(conditional_prob).unsqueeze(-1)],
                     dim=-1)


  def forward(
    self,
    mc_box: torch.Tensor,
    # targets: Optional[torch.Tensor] = None,
    is_training: bool = True,
    batch_num: Optional[int] = None
  ) -> Tuple[torch.Tensor, None]:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim

    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    # Get intersection
    batch_size = mc_box.data.size()[0]
    # Expand both mention&context and type boxes to the shape of batch_size x
    # num_types x box_embedding_dim. (torch.expand doesn't use extra memory.)
    if self.use_gumbel_baysian:  # Gumbel box
      min_point = torch.stack(
        [mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
         type_box.z.unsqueeze(0).expand(batch_size, -1, -1)])
      min_point = torch.max(
        self.gumbel_beta * torch.logsumexp(min_point / self.gumbel_beta, 0),
        torch.max(min_point, 0)[0])

      max_point = torch.stack([
        mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
        type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)])
      max_point = torch.min(
        -self.gumbel_beta * torch.logsumexp(-max_point / self.gumbel_beta, 0),
        torch.min(max_point, 0)[0])

    else:
      min_point = torch.max(
        torch.stack([
          mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

      max_point = torch.min(
        torch.stack([
          mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

    # Get soft volume
    # batch_size x num types
    # Compute the volume of the intersection
    vol1 = self.log_soft_volume(min_point,
                                max_point,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Compute  the volume of the mention&context box
    vol2 = self.log_soft_volume(mc_box.z,
                                mc_box.Z,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Returns log probs
    log_probs = vol1 - vol2.unsqueeze(-1)

    # Clip values > 1. for numerical stability.
    if (log_probs > 0.0).any():
      print("WARNING: Clipping log probability since it's grater than 0.")
      log_probs[log_probs > 0.0] = 0.0

    # if is_training and targets is not None and self.n_negatives > 0:
    #   pos_idx = torch.where(targets.sum(dim=0) > 0.)[0]
    #   neg_idx = torch.where(targets.sum(dim=0) == 0.)[0]

    #   if self.n_negatives < neg_idx.size()[0]:
    #     neg_idx = neg_idx[torch.randperm(len(neg_idx))[:self.n_negatives]]
    #     log_probs_pos = log_probs[:, pos_idx]
    #     log_probs_neg = log_probs[:, neg_idx]
    #     _log_probs = torch.cat([log_probs_pos, log_probs_neg], dim=-1)
    #     _targets = torch.cat([targets[:, pos_idx], targets[:, neg_idx]], dim=-1)
    #     _weights = None
    #     if self.neg_temp > 0.0:
    #       _neg_logits = log_probs_neg - log1mexp(log_probs_neg)
    #       _neg_weights = F.softmax(_neg_logits * self.neg_temp, dim=-1)
    #       _pos_weights = torch.ones_like(log_probs_pos,
    #                                      device=self.box_embeddings.weight.device)
    #       _weights = torch.cat([_pos_weights, _neg_weights], dim=-1)
    #     return _log_probs, _weights, _targets
    #   else:
    #     return log_probs, None, targets
    # elif is_training and targets is not None and self.n_negatives <= 0:
    #   return log_probs, None, targets
    # else:
    #   return log_probs, None, targets
    
    return log_probs

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

