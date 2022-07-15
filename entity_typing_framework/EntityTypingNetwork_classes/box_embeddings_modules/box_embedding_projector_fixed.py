from typing import Optional
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxDecoder, BoxEmbeddingProjector
import torch.nn as nn
import torch
from typing import Optional, Tuple
import torch.nn.functional as F

class BoxEmbeddingProjectorFixed(BoxEmbeddingProjector):
  def __init__(self, type_number, input_dim, box_decoder_params, projection_network_params, box_embeddings_dimension=109, **kwargs) -> None:
    super().__init__(type_number, input_dim, box_decoder_params, projection_network_params, box_embeddings_dimension, **kwargs)
    self.box_decoder = BoxDecoderFixed(num_embeddings = type_number, embedding_dim = box_embeddings_dimension, **box_decoder_params)

class BoxDecoderFixed(BoxDecoder):
  def __init__(self, num_embeddings: int, 
                      embedding_dim: int = 109, 
                      box_type: str = 'CenterSigmoidBoxTensor', 
                      padding_idx: Optional[int] = None, 
                      max_norm: Optional[float] = None, 
                      norm_type: float = 2, 
                      scale_grad_by_freq: bool = False, 
                      sparse: bool = False, 
                      _weight: Optional[torch.Tensor] = None, 
                      init_interval_delta: float = 0.5, 
                      init_interval_center: float = 0.01, 
                      inv_softplus_temp = 1.0, 
                      softplus_scale: float = 1, 
                      n_negatives: int = 0, 
                      neg_temp: float = 0, 
                      box_offset: float = 0.5, 
                      pretrained_box: Optional[torch.Tensor] = None, 
                      use_gumbel_baysian: bool = False, 
                      gumbel_beta = 1.0):
     
     super().__init__(num_embeddings, embedding_dim, box_type, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, init_interval_delta, init_interval_center, inv_softplus_temp, softplus_scale, n_negatives, neg_temp, box_offset, pretrained_box, use_gumbel_baysian, gumbel_beta)
     self.softplus = LearnedSoftPlus(beta=self.inv_softplus_temp)
    #  self.softplus = LearnedSoftPlusWithoutTemp() # with unified beta & temp
     self.gumbel_beta = nn.Parameter(torch.tensor(gumbel_beta))

  def log_soft_volume(
    self,
    z: torch.Tensor,
    Z: torch.Tensor,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta = None
    ) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if gumbel_beta <= 0.:
      return (torch.sum(
        torch.log(self.softplus(Z - z).clamp_min(eps)),
        # torch.log(self.softplus(Z - z, beta=gumbel_beta).clamp_min(eps)), # with unified beta & temp
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
      return (torch.sum(
        torch.log(
          # self.softplus(Z - z - 2 * self.euler_gamma * gumbel_beta).clamp_min(
          self.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma).clamp_min(
          # self.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma, beta=gumbel_beta).clamp_min( # with unified beta & temp
            eps)),
          dim=-1) + torch.log(s))

class BoxEmbeddingProjectorFixedConstrained(BoxEmbeddingProjector):
  def __init__(self, type_number, input_dim, box_decoder_params, projection_network_params, box_embeddings_dimension=109, **kwargs) -> None:
    super().__init__(type_number, input_dim, box_decoder_params, projection_network_params, box_embeddings_dimension, **kwargs)
    self.box_decoder = BoxDecoderFixedConstrained(num_embeddings = type_number, embedding_dim = box_embeddings_dimension, **box_decoder_params)


class BoxDecoderFixedConstrained(BoxDecoder):
  def __init__(self, num_embeddings: int, 
                      embedding_dim: int = 109, 
                      box_type: str = 'CenterSigmoidBoxTensor', 
                      padding_idx: Optional[int] = None, 
                      max_norm: Optional[float] = None, 
                      norm_type: float = 2, 
                      scale_grad_by_freq: bool = False, 
                      sparse: bool = False, 
                      _weight: Optional[torch.Tensor] = None, 
                      init_interval_delta: float = 0.5, 
                      init_interval_center: float = 0.01, 
                      inv_softplus_temp = 1.0, 
                      softplus_scale: float = 1, 
                      n_negatives: int = 0, 
                      neg_temp: float = 0, 
                      box_offset: float = 0.5, 
                      pretrained_box: Optional[torch.Tensor] = None, 
                      use_gumbel_baysian: bool = False, 
                      gumbel_beta = 1.0):
     
     super().__init__(num_embeddings, embedding_dim, box_type, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, init_interval_delta, init_interval_center, inv_softplus_temp, softplus_scale, n_negatives, neg_temp, box_offset, pretrained_box, use_gumbel_baysian, gumbel_beta)
    #  self.softplus = LearnedSoftPlus(beta=self.inv_softplus_temp)
    #  self.softplus = LearnedSoftPlusWithoutTemp() # with unified beta & temp
    #  self.gumbel_beta = gumbel_beta))

  def log_soft_volume(
    self,
    z: torch.Tensor,
    Z: torch.Tensor,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta = None
    ) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if gumbel_beta <= 0.:
      return (torch.sum(
        torch.log(self.softplus(Z - z).clamp_min(eps)),
        # torch.log(self.softplus(Z - z, beta=gumbel_beta).clamp_min(eps)), # with unified beta & temp
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
      return (torch.sum(
        torch.log(
          # self.softplus(Z - z - 2 * self.euler_gamma * gumbel_beta).clamp_min(
          # self.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma).clamp_min(
          F.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma, beta=self.inv_softplus_temp).clamp_min( # with unified beta & temp
            eps)),
          dim=-1) + torch.log(s))

  def forward(self, mc_box: torch.Tensor, is_training: bool = True, batch_num: Optional[int] = None):
    log_probs = super().forward(mc_box, is_training, batch_num)
    
    constrained_log_probs = torch.clamp(log_probs, -6.)

    return constrained_log_probs

class BoxDecoderFixedWithDifferentMCBoxVolume(BoxDecoder):
  def get_gumbel_min_max(self, boxes_1, boxes_2, batch_size, num_embeddings):
    if self.use_gumbel_baysian:  # Gumbel box
        min_point = torch.stack(
          [boxes_1.z.unsqueeze(1).expand(-1, num_embeddings, -1),
          boxes_2.z.unsqueeze(0).expand(batch_size, -1, -1)])
        min_point = torch.max(
          self.gumbel_beta * torch.logsumexp(min_point / self.gumbel_beta, 0),
          torch.max(min_point, 0)[0])

        max_point = torch.stack([
          boxes_1.Z.unsqueeze(1).expand(-1, num_embeddings, -1),
          boxes_2.Z.unsqueeze(0).expand(batch_size, -1, -1)])
        max_point = torch.min(
          -self.gumbel_beta * torch.logsumexp(-max_point / self.gumbel_beta, 0),
          torch.min(max_point, 0)[0])

    else:
      min_point = torch.max(
        torch.stack([
          boxes_1.z.unsqueeze(1).expand(-1, num_embeddings, -1),
          boxes_2.z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

      max_point = torch.min(
        torch.stack([
          boxes_1.Z.unsqueeze(1).expand(-1, num_embeddings, -1),
          boxes_2.Z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]
    return min_point, max_point

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
      min_intersections, max_intersections = self.get_gumbel_min_max(mc_box, 
                                                                      type_box, 
                                                                      batch_size = batch_size, 
                                                                      num_embeddings = self.num_embeddings)
      min_mcs, max_mcs = self.get_gumbel_min_max(mc_box, 
                                                  mc_box, 
                                                  batch_size = batch_size, 
                                                  num_embeddings = batch_size)

      min_diag_x = min_mcs[:, :, 0].diag()
      min_diag_y = min_mcs[:, :, 1].diag()
      
      min_mcs = torch.stack((min_diag_x, min_diag_y)).T

      max_diag_x = max_mcs[:, :, 0].diag()
      max_diag_y = max_mcs[:, :, 1].diag()

      max_mcs = torch.stack((max_diag_x, max_diag_y)).T


      # Get soft volume
      # batch_size x num types
      # Compute the volume of the intersection
      vol1 = self.log_soft_volume(min_intersections,
                                  max_intersections,
                                  temp=self.inv_softplus_temp,
                                  scale=self.softplus_scale,
                                  gumbel_beta=self.gumbel_beta)

      # Compute  the volume of the mention&context box
      vol2 = self.log_soft_volume(min_mcs,
                                  max_mcs,
                                  temp=self.inv_softplus_temp,
                                  scale=self.softplus_scale,
                                  gumbel_beta=self.gumbel_beta)

      # Returns log probs
      log_probs = vol1 - vol2.unsqueeze(-1)

      # Clip values > 1. for numerical stability.
      if (log_probs > 0.0).any():
        print("WARNING: Clipping log probability since it's greater than 0.")
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


class LearnedSoftPlus(torch.nn.Module):
  def __init__(self, beta=1.0, threshold=20):
    super().__init__()
    # keep beta > 0
    self.log_beta = torch.nn.Parameter(torch.tensor(float(beta)).log())
    self.threshold = threshold

  def forward(self, x):
    beta = self.log_beta.exp()
    beta_x = beta * x
    return torch.where(beta_x < self.threshold, torch.log1p(beta_x.exp()) / beta, x)

class LearnedSoftPlusWithoutTemp(torch.nn.Module):
  def __init__(self, threshold=20):
    super().__init__()
    # keep beta > 0
    # self.log_beta = torch.nn.Parameter(torch.tensor(float(beta)).log())
    self.threshold = threshold

  def forward(self, x, beta):
    # beta = beta.exp()
    beta_x = beta * x
    return torch.where(beta_x < self.threshold, torch.log1p(beta_x.exp()) / beta, x)