from entity_typing_framework.main_module.losses_modules import BCELossModule
import torch

class BCENFETCLossModule(BCELossModule):
  def __init__(self, beta, type2id, loss_params, **kwargs) -> None:
    super().__init__(type2id, loss_params, **kwargs)
    self.beta = beta
    
  def create_prior(self, type2id):
    self.type2id = type2id
    # create ancestors map utils
    ancestors_map = {}
    for t in type2id.keys():
      splitted_path = t.replace('/',' /').split(' ')[1:]
      ancestors_map[t] = []
      prev_ancestor = ''
      for ancestor in splitted_path[:-1]:
        ancestor = prev_ancestor + ancestor
        ancestors_map[t].append(ancestor)    
        prev_ancestor = ancestor
    self.ancestors_map = ancestors_map
    
    # create ancestors priors to be summed when computing loss
    n_types = len(self.type2id)
    prior = torch.zeros((n_types, n_types))
    for t, ancestors in self.ancestors_map.items():
      # assign 1 to the diagonal
      idx_t = self.type2id[t]
      prior[idx_t, idx_t] = 1.0
      # assign weight self.beta to the ancestors
      for ancestor in ancestors:
        idx_ancestor =  self.type2id[ancestor]
        prior[idx_t,idx_ancestor] = self.beta
    
    self.ancestors_prior = prior.T


  def compute_loss(self, encoded_input, type_representation):
    if encoded_input.device.type == 'cuda' and self.ancestors_prior.device.type != 'cuda':
      self.ancestors_prior = self.ancestors_prior.cuda()
    # inject hierarchy
    encoded_input = torch.matmul(encoded_input, self.ancestors_prior)
    # re-normalize probabilities
    encoded_input = torch.clip(encoded_input, 1e-10, 1.0)
    return self.loss(encoded_input, type_representation)

class BCENFETCCustomLossModule(BCENFETCLossModule):
  def compute_loss(self, encoded_input, type_representation):
    if encoded_input.device.type == 'cuda' and self.ancestors_prior.device.type != 'cuda':
      self.ancestors_prior = self.ancestors_prior.cuda()
    # transform the probabilities into preactivations
    encoded_input = self.inverse_sigmoid(encoded_input)
    # inject hierarchy
    encoded_input = torch.matmul(encoded_input, self.ancestors_prior)
    # re-normalize probabilities
    encoded_input = torch.sigmoid(encoded_input)
    return self.loss(encoded_input, type_representation)

  # TODO: utils file with this kind of functions
  def inverse_sigmoid(self, x):
    # from https://discuss.pytorch.org/t/inverse-of-sigmoid-in-pytorch/14215
    return torch.log(x/(1 - x)) 