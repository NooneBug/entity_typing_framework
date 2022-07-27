
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
import torch

# NOTE: wrong interpretation of NFETC
class NFETCClassifier(Classifier):
  def __init__(self, beta=.35, **kwargs):
    super().__init__(**kwargs)
    # self.beta = torch.nn.Parameter(torch.tensor(beta))
    self.beta = beta
    self.ancestors_map = self.get_ancestors_map()

  def get_ancestors_map(self):
    ancestors_map = {}

    for t in self.type2id.keys():
      splitted_path = t.replace('/',' /').split(' ')[1:]
      ancestors_map[t] = []
      prev_ancestor = ''
      for ancestor in splitted_path[:-1]:
        ancestor = prev_ancestor + ancestor
        ancestors_map[t].append(ancestor)    
        prev_ancestor = ancestor

    return ancestors_map

  # NOTE: should use matmul instead of cascading
  def forward(self, encoded_input):
    projected_input = super().forward(encoded_input)
    # transform input to modify preactivations
    projected_input = self.inverse_sigmoid(projected_input)
    # modify predictions according to the hierarchy
    for t, ancestors in self.ancestors_map.items():
      idx_t = self.type2id[t]
      for ancestor in ancestors:
        idx_ancestor =  self.type2id[ancestor]
        projected_input[:,idx_t] = projected_input[:,idx_t] + self.beta * projected_input[:,idx_ancestor]
    # re-normalize output
    return torch.sigmoid(projected_input)
    
  
  def inverse_sigmoid(self, x):
    # from https://discuss.pytorch.org/t/inverse-of-sigmoid-in-pytorch/14215
    return torch.log(x/(1 - x)) 
