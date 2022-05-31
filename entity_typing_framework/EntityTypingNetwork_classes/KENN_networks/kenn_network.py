from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForIncrementalTraining
from pytorch_lightning import LightningModule
import torch
from torch.nn import Sigmoid
from tqdm import tqdm
import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils

import sys
sys.path.append('./')
from kenn.parsers import unary_parser

class KENNClassifier(LightningModule):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down', **kwargs):
    # explicit super call to avoid multiple inheritance problems in KENNClassifier's subclasses
    super(LightningModule, self).__init__()
    # classifier
    self.classifier = Classifier(**kwargs)

    if not clause_file_path:
      clause_file_path = 'kenn_tmp/clause_file_path.txt'
      id2type = {v: k for k, v in self.classifier.type2id.items()}
      # generate and save KENN clauses
      self.automatic_build_clauses(types_list = [id2type[idx] for idx in range(len(id2type))], clause_file_path=clause_file_path,
                                  learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode)
    
    # instance KENN layer by reading the clauses from the clause_file_path file 
    self.ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )
    self.sig = Sigmoid()

  def forward(self, input_representation):
    prekenn = self.classifier(input_representation=input_representation)
    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(prekenn), self.sig(postkenn)

  def automatic_build_clauses(self, types_list, clause_file_path = None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down'):
    # generate and save KENN clauses
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints(types_list, kb_mode, clause_file_path, cw)

  def get_state_dict(self, light=True):
    return self.state_dict()

class KENNClassifierForIncrementalTraining(KENNClassifier, ClassifierForIncrementalTraining):
  def __init__(self, clause_file_path=None, learnable_clause_weight=False, clause_weight=0.5, kb_mode='top_down', **kwargs):
    kwargs_pretraining = self.get_kwargs_pretrained_projector(**kwargs)
    KENNClassifier.__init__(self, 
                            clause_file_path=clause_file_path,
                            learnable_clause_weight=learnable_clause_weight,
                            clause_weight=clause_weight,
                            kb_mode=kb_mode,
                            **kwargs_pretraining)
    kwargs_additional_classifier = self.get_kwargs_additional_classifier(**kwargs)
    self.additional_classifier = Classifier(**kwargs_additional_classifier)
    
    # prepare additional Knowledge Enhancer
    type2id = kwargs['type2id']
    new_type_number = len(kwargs['type2id']) - kwargs['type_number']
    all_types = list(type2id.keys())
    new_types = all_types[-new_type_number:]
    clause_file_path = 'kenn_tmp/incremental_clause_file_path.txt'
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints_incremental(all_types=all_types,
                                                new_types=new_types,
                                                filepath=clause_file_path,
                                                weight=cw)
    self.additional_ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )

  def forward(self, input_representation):
    # predict pretraining types
    # prekenn_pretrain = self.classifier(input_representation)
    # postkenn_pretrain = self.ke(prekenn_pretrain)[0]
    _, postkenn_pretrain = super().forward(input_representation)
    # predict incremental types
    prekenn_incremental_projected_representation = self.classifier.project_input(input_representation)
    prekenn_incremental = self.additional_classifier.classify(prekenn_incremental_projected_representation)
    prekenn_all_types = torch.concat((postkenn_pretrain, prekenn_incremental), dim=1) # why postkenn_pretrain and not prekenn_pretrain? Because it is stacked...
    postkenn_incremental = self.additional_ke(prekenn_all_types)[0][:,-self.additional_classifier.type_number:]
    
    # assemble final prediction
    postkenn_all_types = torch.concat((postkenn_pretrain, postkenn_incremental), dim=1)

    return self.sig(prekenn_all_types), self.sig(postkenn_all_types)

  def freeze_pretraining(self):
    self.freeze()
    self.additional_classifier.unfreeze()
    self.additional_ke.unfreeze()

