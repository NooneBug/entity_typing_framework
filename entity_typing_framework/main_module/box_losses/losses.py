from typing import Optional
import torch.nn as nn
import torch
import math 

# class BCELossForET(LossModule):
#     def __init__(self, name, **kwargs) -> None:
#         super().__init__()
#         self.loss = BCELoss(**kwargs)
    
#     def compute_loss(self, encoded_input, type_representation):
#         return self.loss(encoded_input, type_representation)

# class BoxEmbeddingLogProbBCELoss(Loss):
#     def __init__(self, name):
#         super().__init__()
#         self.loss_func = BCEWithLogProbLoss()

#     def compute_loss(self,
#                     logits: torch.Tensor,
#                     targets: torch.Tensor,
#                     ) -> torch.Tensor:

#         loss = self.loss_func(logits, targets)
#         return loss

_log1mexp_switch = math.log(0.5)

def log1mexp(x: torch.Tensor,
             split_point=_log1mexp_switch,
             exp_zero_eps=1e-7) -> torch.Tensor:
  """
  Computes log(1 - exp(x)).
  Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).
  = log1p(-exp(x)) when x <= log(1/2)
  or
  = log(-expm1(x)) when log(1/2) < x <= 0
  For details, see
  https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
  """
  logexpm1_switch = x > split_point
  Z = torch.zeros_like(x)
  # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
  # ie. p=1
  logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
  # hack the backward pass
  # if expm1(x) gets very close to zero, then the grad log() will produce inf
  # and inf*0 = nan. Hence clip the grad so that it does not produce inf
  logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
  Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
  #Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
  Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

  return Z

class BCEWithLogProbLoss(nn.BCELoss):

  def __init__(self, **kwargs) -> None:
    if 'name' in kwargs:
      kwargs.pop('name')
    super().__init__(**kwargs)

  def forward(self,
                            input: torch.Tensor,
                            target: torch.Tensor,
                            weight: Optional[torch.Tensor] = None,
                            reduction: str = 'mean') -> torch.Tensor:
    """Computes binary cross entropy.
    This function takes log probability and computes binary cross entropy.
    Args:
      input: Torch float tensor. Log probability. Same shape as `target`.
      target: Torch float tensor. Binary labels. Same shape as `input`.
      weight: Torch float tensor. Scaling loss if this is specified.
      reduction: Reduction method. 'mean' by default.
    """
    loss = -target * input - (1 - target) * log1mexp(input)

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()

