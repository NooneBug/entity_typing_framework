from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
import torch
from torch.nn import Sigmoid
from tqdm import tqdm
import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils
import sys
sys.path.append('./')
from kenn.parsers import unary_parser

class KENNClassifier(Classifier):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down', **kwargs):
    super().__init__(**kwargs)

    if not clause_file_path:
      clause_file_path = 'kenn_tmp/clause_file_path.txt'
      id2type = {v: k for k, v in self.type2id.items()}
      self.automatic_build_clauses(types_list = [id2type[idx] for idx in range(len(id2type))], clause_file_path=clause_file_path,
                                  learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode)
    
    # KENN layer
    self.ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x # linear activation
                          )
    self.sig = Sigmoid()

  def forward(self, input_representation):
    prekenn = super().forward(input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(postkenn)
  
  def automatic_build_clauses(self, types_list, clause_file_path = None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down'):
    # generate KENN clauses
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints(types_list, kb_mode, clause_file_path, cw)

class KENNClassifierMultiloss(KENNClassifier):
  def __init__(self, clause_file_path=None, **kwargs):
    super().__init__(clause_file_path, **kwargs)
  
  def forward(self, input_representation):
    prekenn = Classifier.forward(self, input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(prekenn), self.sig(postkenn)
