from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
import torch
from torch.nn import Sigmoid
from tqdm import tqdm

import sys
sys.path.append('./')
from kenn.parsers import unary_parser

class KENNClassifier(Classifier):
  def __init__(self, clause_file_path, save_training_data=False, **kwargs):
    super().__init__(**kwargs)
    # KENN layer
    self.ke = unary_parser(knowledge_file=clause_file_path,
                           activation=lambda x: x, # linear activation
                           save_training_data=save_training_data)
    self.sig = Sigmoid()

  def forward(self, input_representation):
    prekenn = super().forward(input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(postkenn)

class KENNClassifierMultiloss(KENNClassifier):
  def __init__(self, clause_file_path, save_training_data=False, **kwargs):
    super().__init__(clause_file_path, save_training_data, **kwargs)
  
  def forward(self, input_representation):
    prekenn = Classifier.forward(self, input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(prekenn), self.sig(postkenn)
